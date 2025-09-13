import torch.nn as nn
from .vqvae import VQVAE
# from .discrim import DinoDiscV2
from .unitok import UniTok
from ..utils.config import Args


def build_vae(args: Args):
    vae = VQVAE(args).to(args.device)
    init_weights(vae.encoder, args.vae_init)
    init_weights(vae.decoder, args.vae_init)
    init_weights(vae.quant_proj, args.vae_init)
    init_weights(vae.post_quant_proj, args.vae_init)
    vae.quantize.init_vocab(args.vocab_init)
    return vae


def build_discriminator(args: Args):
    return None


def build_unitok(args: Args):
    model = UniTok(args).to(args.device)
    # init_weights(model.encoder, args.vae_init)
    init_weights(model.decoder, args.vae_init)
    init_weights(model.quant_proj, args.vae_init)
    init_weights(model.post_quant_proj, args.vae_init)
    model.quantizer.init_vocab(args.vocab_init)
    return model


def init_weights(model, conv_std_or_gain):
    print(f'[init_weights] {type(model).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if conv_std_or_gain > 0:
                nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
            else:
                nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.)
