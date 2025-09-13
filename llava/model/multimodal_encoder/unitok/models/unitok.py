from llava.model.multimodal_encoder.unitok.utils.config import Args
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from .vitamin import GeGluMlp, ViTaminDecoder
from .quant import VectorQuantizerM
from .vqvae import AttnProjection


class UniTok(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_query = args.num_query

        self.encoder = timm.create_model(
            args.model,
            patch_size=1,
            fc_norm=False,
            drop_rate=0.0,
            num_classes=0,
            global_pool='',
            pos_embed='none',
            class_token=False,
            mlp_layer=GeGluMlp,
            reg_tokens=args.num_query,
            img_size=args.img_size,
            drop_path_rate=args.drop_path,
        )
        self.encoder.pos_embed = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim), requires_grad=False)

        if args.quant_proj == 'linear':
            self.quant_proj = nn.Linear(self.encoder.embed_dim, args.vocab_width)
        elif args.quant_proj == 'attn':
            self.quant_proj = AttnProjection(self.encoder.embed_dim, args.vocab_width, self.encoder.embed_dim // args.vocab_width)
        else:
            raise NotImplementedError

        self.quantizer = VectorQuantizerM(
            vocab_size=args.vocab_size,
            vocab_width=args.vocab_width,
            beta=args.vq_beta,
            use_entropy_loss=args.le > 0,
            entropy_temp=args.e_temp,
            num_codebooks=args.num_codebooks,
        )

        if args.quant_proj == 'linear':
            self.post_quant_proj = nn.Linear(args.vocab_width, self.encoder.embed_dim)
        elif args.quant_proj == 'attn':
            self.post_quant_proj = AttnProjection(args.vocab_width, self.encoder.embed_dim, self.encoder.embed_dim // args.vocab_width)
        else:
            raise NotImplementedError

        self.decoder = ViTaminDecoder(
            args.model,
            num_query=args.num_query,
            img_size=args.img_size,
            drop_path=args.drop_path,
            grad_ckpt=args.grad_ckpt,
        )

        text_cfg = {
            "width": args.text_width,
            "heads": args.text_heads,
            "layers": args.text_layers,
            "vocab_size": args.text_vocab_size,
            "context_length": args.text_context_length,
        }
        from open_clip.model import _build_text_tower
        self.text_encoder = _build_text_tower(args.embed_dim, text_cfg)

        self.fc_norm = nn.LayerNorm(self.encoder.embed_dim, eps=1e-6)
        self.projection = nn.Linear(self.encoder.embed_dim, args.embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.context_length = self.text_encoder.context_length
        self.vocab_size = self.text_encoder.vocab_size
        self.maybe_record_function = nullcontext

        self.text_no_grad = False
        self.encoder.set_grad_checkpointing(args.grad_ckpt)
        self.text_encoder.set_grad_checkpointing(args.grad_ckpt)

    def forward(self, img, vae_bs, text=None, ret_usages=False):
        img_tokens = self.encoder(img).float()
        with torch.cuda.amp.autocast(enabled=False):
            img_tokens = torch.utils.checkpoint.checkpoint(self.quant_proj, img_tokens, use_reentrant=False)
            img_tokens, vq_loss, entropy_loss, usages = self.quantizer(img_tokens)
            img_tokens = torch.utils.checkpoint.checkpoint(self.post_quant_proj, img_tokens, use_reentrant=False)
        img_rec = self.decoder(img_tokens[:vae_bs]).float()

        clip_visual = img_tokens.mean(dim=1)
        clip_visual = self.projection(self.fc_norm(clip_visual))
        clip_visual = F.normalize(clip_visual, dim=-1)
        if text is not None:
            clip_text = self.text_encoder(text)
            clip_text = F.normalize(clip_text, dim=-1)
        else:
            clip_text = None

        output_dict = {
            "img_rec": img_rec,
            "vq_loss": vq_loss,
            "entropy_loss": entropy_loss,
            "codebook_usages": usages,
            "clip_image_features": clip_visual,
            "clip_text_features": clip_text,
            "logit_scale": self.logit_scale.exp()
        }
        return output_dict

    def encode_image(self, image, normalize: bool = False):
        img_tokens = self.encoder(image)
        img_tokens = self.quant_proj(img_tokens)
        img_indices = self.quantizer.f_to_idx(img_tokens)
        img_tokens = self.quantizer.idx_to_f(img_indices)
        img_tokens = self.post_quant_proj(img_tokens)
        features = img_tokens.mean(dim=1)
        features = self.projection(self.fc_norm(features))
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text_encoder(text)
        return F.normalize(features, dim=-1) if normalize else features

    def img_to_idx(self, img):
        features = self.encoder(img)#.float()
        
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            features = self.quant_proj(features)
        return self.quantizer.f_to_idx(features)

    def idx_to_img(self, indices):
        features = self.quantizer.idx_to_f(indices)
        features = self.post_quant_proj(features)
        img = self.decoder(features).clamp_(-1, 1)
        return img

    def img_to_reconstructed_img(self, image) -> torch.Tensor:
        img_tokens = self.encoder(image)
        img_tokens = self.quant_proj(img_tokens)
        img_tokens, _, _, _ = self.quantizer(img_tokens)
        img_tokens = self.post_quant_proj(img_tokens)
        img_rec = self.decoder(img_tokens).clamp_(-1, 1)
        return img_rec

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True, unlock_text_proj=False):
        self.text.lock(unlocked_layers, freeze_layer_norm, unlock_text_proj)
        self.text_no_grad = True


class UniTokEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(8192+256, config.d_model_gen) # hack
        self.embedding_cont = nn.Linear(4096, config.d_model_gen, bias=False)
        self.norm = nn.RMSNorm(config.d_model_gen, eps=1e-06, elementwise_affine=True)
        self.out_linear = nn.Linear(config.d_model_gen, config.d_model_gen, bias=False)
        # self.quantizer = None
        quant_dim = 64
        self.up_proj = nn.Sequential(
            nn.Linear(quant_dim,config.d_model_gen),
        )
        self.mm_sub_mask = getattr(config,'mm_submask',False)
        # if self.mm_sub_mask:
        #     self.mask_embedding_v2 = nn.Embedding(16,8)

    def mask_embedding_v2(self,device):
        # hack
        embeddings = self.embedding(torch.tensor([8200]).to(device))[:,:64] # 1, 64
        embeddings = embeddings.view(8,8) # 8 X 8
        return embeddings

    def forward(self,indices,unitok,mask_id = 8193):
        indices = indices.clone()
        assert len(indices.shape) == 3, f"Expected indices to be of shape (batch_size, num_codebooks, seq_length), got {indices.shape}"
        
        raw_is_mask = indices == mask_id # B X 8 X L
        is_mask = raw_is_mask.all(1) # B X  L
        indices[indices == mask_id] = 0
        features = unitok.quantizer.idx_to_f(indices) # 1 L 64
        if self.mm_sub_mask:
            _b,_l,_ = features.shape
            features = features.view(_b,_l,8,-1) # B L NUM_CODEBOOK CODEBOOK_FEAT
            nosie_embeddings = self.mask_embedding_v2(features.device) # 8 (N_COEEBOOK) X 8
            features = torch.where(raw_is_mask.permute(0,2,1).unsqueeze(-1).repeat(1,1,1,8),nosie_embeddings[None,None],features)
            features = features.view(_b,_l,-1)
        up_proj = self.up_proj(features) # 1 L D
        if not self.mm_sub_mask:
            noise_embeddings = self.embedding(torch.tensor([mask_id],device=indices.device)) # 1 D
            up_proj_with_noise = torch.where(is_mask.unsqueeze(-1), noise_embeddings.unsqueeze(1), up_proj) # 1 L D
        else:
            noise_embeddings = self.embedding(torch.tensor([mask_id],device=indices.device)) # 1 D
            up_proj_with_noise = torch.where(is_mask.unsqueeze(-1), noise_embeddings.unsqueeze(1), up_proj) # 1 L D
        x = self.norm(up_proj_with_noise)
        x = self.out_linear(x)
        return x


def build_unitok(path):
    ckpt_path = '/mnt/localssd/unitok_tokenizer/unitok_tokenizer.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    unitok_cfg = Args()
    unitok_cfg.load_state_dict(ckpt['args'])
    unitok = UniTok(unitok_cfg)
    unitok.load_state_dict(ckpt['trainer']['unitok'])
    unitok.eval()
    unitok.requires_grad_(False)
    return unitok


if __name__ == '__main__':
    model = timm.create_model(
        'vitamin_base',
        patch_size=1,
        fc_norm=True,
        drop_rate=0.0,
        num_classes=0,
        global_pool='',
        pos_embed='none',
        class_token=False,
        mlp_layer=GeGluMlp,
        reg_tokens=0,
        img_size=256,
        drop_path_rate=0.1,
    )
    model.pos_embed = nn.Parameter(torch.zeros(1, 1, model.embed_dim), requires_grad=False)

    model_dict = model.state_dict()
    ckpt_dict = torch.load('ViTamin-B/pytorch_model.bin')
    visual_dict = dict()
    for k, v in ckpt_dict.items():
        if k.startswith('visual.'):
            if 'head' in k or 'pos_embed' in k:
                continue
            new_k = k.replace('visual.trunk.', '')
            visual_dict[new_k] = v

    model.load_state_dict(visual_dict, strict=False)
    print(set(model_dict.keys()) - set(visual_dict.keys()))
    print(set(visual_dict.keys() - set(model_dict.keys())))

