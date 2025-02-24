import torch
import torch.nn as nn
import re

from .d_abs import DAbstractor


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def build_mlp_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_dabstractor_projector(config):
    return DAbstractor(config)

# def build_cross_projector(conifg):
#     return CrossProjector(config)

# class CrossDabstractor(conifg):
#     def __init__(self, config):
#         super().__init__()
#         self.dabstractor = DAbstractor(config)
#         self.cross_attn = CrossProjector(config)
    
#     def forward(self, image_features, semantic_features):
#         attn_output, attn_weights = self.cross_attn(
#             hidden_states=semantic_features,
#             key_value_states=image_features,
#             attention_mask=None,
#             layer_head_mask=None,
#             output_attentions=True,
#         )
#         return self.d_abstractor(attn_output)

# def build_cross_dabstractor_projector(config):
#     return CrossDabstractor(config)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    # if projector_type == 'cross_dabstractor':
    #     return build_cross_dabstractor_projector(config)
    
    if projector_type == "dasbtractor":
        return build_dabstractor_projector(config)

    # if projector_type == 'cross':
    #     return build_cross_projector(config)

    return build_mlp_projector(config, delay_load=delay_load)
    
