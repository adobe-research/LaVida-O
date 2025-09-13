import torch
import torch.nn.functional as F
import pytorch_lightning as pl
# import lightning as L

# from main import instantiate_from_config
from collections import OrderedDict
from contextlib import contextmanager

from .modules.diffusionmodules.model import Encoder, Decoder
from .modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from .modules.vqvae.quantize import IndexPropagationQuantize
#from .modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from .modules.ema import LitEma
from torch import nn
import numpy as np
class VQModel(nn.Module):
    def __init__(self,
                ddconfig,
                lossconfig,
                #Quantize Related
                n_embed,
                embed_dim,
                ckpt_path = None,
                ignore_keys = [],
                image_key = "image",
                colorize_nlabels = None,
                monitor = None,
                remap = None,
                sane_index_shape = False,  # tell vector quantizer to return indices as bhw
                learning_rate = None,
                l2_normalize = False,
                ### scheduler config
                warmup_epochs = 0,  # warmup epochs
                scheduler_type = "None",
                min_learning_rate = 0,
                gradient_clip_val = 0,
                resume_lr = None,
                lr_drop_epoch = None,
                lr_drop_rate = 0.1,
                use_ema = False,
                stage = None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = None #  instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape, l2_normalize=l2_normalize)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.stage = stage
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = use_ema
        if self.use_ema and stage is None: #no need to construct EMA when training Transformer
            self.model_ema = LitEma(self)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)

        self.learning_rate = learning_rate
        self.automatic_optimization = False
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.gradient_clip_val = gradient_clip_val
        self.resume_lr = resume_lr
        self.lr_drop_epoch = lr_drop_epoch
        self.lr_drop_rate = lr_drop_rate

        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().load_state_dict(*args, strict=strict)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}

    def init_from_ckpt(self, path, ignore_keys=list(), stage="transformer"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue
                    if "embedding" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "quant" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
                    elif "embedding" in k:
                        new_params[k] = v
                    elif "quant" in k:
                        new_params[k] = v              
        missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x.float()

    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                opt_gen_param_group["lr"] = self.resume_lr
                opt_disc_param_group["lr"] = self.resume_lr

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def on_train_epoch_end(self):
        ### update lr
        self.lr_annealing()

    def lr_annealing(self):
        """
        Perform Lr decay
        """
        if self.lr_drop_epoch is not None:
            current_epoch = self.trainer.current_epoch
            if (current_epoch + 1) in self.lr_drop_epoch:
                opt_gen, opt_disc = self.optimizers()
                for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                    opt_gen_param_group["lr"] = opt_gen_param_group["lr"] * self.lr_drop_rate
                    opt_disc_param_group["lr"] = opt_disc_param_group["lr"] * self.lr_drop_rate
    
    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        opt_gen, opt_disc = self.optimizers()
        if self.scheduler_type != "None":
            scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        # original VQGAN first optimizes G, then D. We first optimize D then G, following traditional GAN
        # optimize discriminator
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)


        # optimize generator
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        opt_gen.zero_grad()
        self.manual_backward(aeloss)


        if self.gradient_clip_val > 0: # for cosine similarity
            self.clip_gradients(opt_gen, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm="norm")

        opt_gen.step()
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        if self.scheduler_type != "None":
            scheduler_disc.step()
            scheduler_gen.step()

    def validation_step(self, batch, batch_idx):
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
            return log_dict_ema
        else:
            log_dict = self._validation_step(batch, batch_idx)
            return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        quant, qloss, (_, _, min_encoding_indices) = self.encode(x)
        x_rec = self.decode(quant).clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(qloss, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        raise NotImplementedError("Disabled")

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class IBQ(VQModel):
    def __init__(self,
                ddconfig,
                lossconfig,
                n_embed,
                embed_dim,
                ckpt_path = None,
                ignore_keys = [],
                image_key = "image",
                colorize_nlabels = None,
                monitor = None,
                remap = None,
                sane_index_shape = False,  # tell vector quantizer to return indices as bhw
                learning_rate = None,
                l2_normalize = False,
                ### scheduler config
                warmup_epochs = 0,  # warmup epochs
                scheduler_type = "None",
                min_learning_rate = 0,
                cosine_similarity = False,
                gradient_clip_val = 0,
                use_entropy_loss = False,
                sample_minimization_weight = 1.0,
                batch_maximization_weight = 1.0,
                entropy_temperature = 0.01,
                beta = 0.25,
                lr_drop_epoch = None,
                lr_drop_rate = 0.1,
                resume_lr = None,
                use_ema = False,
                stage = None,
                 ):
        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                        lossconfig,
                        n_embed,
                        embed_dim,
                        ckpt_path=None,
                        ignore_keys=ignore_keys,
                        image_key=image_key,
                        colorize_nlabels = colorize_nlabels,
                        monitor = monitor,
                        remap = remap,
                        sane_index_shape = sane_index_shape,
                        learning_rate = learning_rate,
                        l2_normalize = l2_normalize,
                        warmup_epochs = warmup_epochs,
                        scheduler_type = scheduler_type,
                        min_learning_rate = min_learning_rate,
                        gradient_clip_val = gradient_clip_val,
                        resume_lr = resume_lr,
                        use_ema = use_ema,
                        stage = stage,
                        lr_drop_epoch = lr_drop_epoch,
                        lr_drop_rate = lr_drop_rate
                        )
        self.quantize = IndexPropagationQuantize(n_embed, embed_dim, beta, use_entropy_loss,
                                          remap=remap, cosine_similarity=cosine_similarity,
                                          entropy_temperature=entropy_temperature,
                                          sample_minimization_weight=sample_minimization_weight, batch_maximization_weight=batch_maximization_weight)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)

import yaml
import os

def preprocess_image(image):
    
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    image = (image/127.5 - 1.0).astype(np.float32)
    image = torch.tensor(image).permute(2,0,1).float()
    return image

def build_ibq():
    dir_name = os.path.dirname(__file__)
    ckpt = '/mnt/localssd/IBQ-Tokenizer-16384-Pretrain/IBQ_pretrain_16384.ckpt'
    config_path = os.path.join(dir_name, "/sensei-fs-3/users/shufanl/LaViDa/llava/model/multimodal_encoder/ibq/pretrain_ibqgan_16384.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = config['model']['init_args']
    model = IBQ(**args)
    state_dict = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state_dict['state_dict'])
    del state_dict
    return model, preprocess_image

class IBQProcessor:
    def __init__(self):
        pass
    
    def preprocess(self, image):
        return preprocess_image(image)

if __name__ == "__main__":
    '''
    python -m llava.model.multimodal_encoder.ibq.ibqgan
    '''
    from PIL import Image
    img = Image.open('/sensei-fs-3/users/shufanl/LaViDa/images/port.png').convert('RGB')#.resize((1024,716))#.resize((512,512)
    img = img.resize((256,256))

    model,processor = build_ibq()
    model.cuda()
    
    model.requires_grad_(False)
    img = preprocess_image(img)
    
    img = img.unsqueeze(0).cuda()  # (1, C, H, W)
    breakpoint()
    quant, qloss, (_, _, indices) = model.encode(img)
    reconstructed_images = model.decode(quant)
    reconstructed_images = reconstructed_images.clamp(-1, 1)
    reconstructed_images = (reconstructed_images + 1) / 2
    reconstructed_images = (reconstructed_images * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # (N, H, W, C)
    reconstructed_image = reconstructed_images[0]
    image = Image.fromarray(reconstructed_image)
    image.save('a.jpg')