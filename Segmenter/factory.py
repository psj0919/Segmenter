from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from Segmenter.vit import VisionTransformer
from Segmenter.utils import checkpoint_filter_fn
from Segmenter.decoder import DecoderLinear
from Segmenter.decoder import MaskTransformer
from Segmenter.Segmenter import Segmenter
from timm.models.vision_transformer import _load_weights

@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 224, 224),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg["model"]["backbone"]['name']

    normalization = model_cfg['model']['backbone']["normalization"]
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg['model']['backbone']['d_ff'] = mlp_expansion_ratio * model_cfg['model']['backbone']["d_model"]


    model = VisionTransformer(
        model_cfg['model']['backbone']['image_size'],
        model_cfg['model']['backbone']['patch_size'],
        model_cfg['model']['backbone']['n_layers'],
        model_cfg['model']['backbone']['d_model'],
        model_cfg['model']['backbone']['d_ff'],
        model_cfg['model']['backbone']['n_heads'],
        model_cfg['model']['backbone']['n_cls'],
    )

    return model

def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg['name']
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(
            decoder_cfg['n_cls'],
            decoder_cfg['patch_size'],
            decoder_cfg['d_encoder'],
            decoder_cfg['n_layers'],
            decoder_cfg['n_heads'],
            decoder_cfg['d_model'],
            decoder_cfg['d_ff'],
            decoder_cfg['drop_path_rate'],
            decoder_cfg['dropout'],
        )
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg["decoder"]
    # decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg['model']['backbone']["n_cls"])

    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location='cpu')
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant