import torch
import torch.nn as nn
import re


from transformers import BartConfig, BartDecoderLayer

def build_semantic_projector(config, **kwargs):
    return SemanticProjector(config)

class SemanticProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.d_abstractor = DAbstractor(config, config.num_input_tokens)
        self.cross_attn = BartDecoderLayer(BartConfig(**config.bart_decoder_layer))
        self.linear = nn.Linear(config.semantic_dim, config.image_feature_dim)

        #initialize the weights
        self.linear.weight.data.normal_(mean=0.0, std=config.init_std)

    def forward(self, image_features, semantic_features):
        semantic_features = self.linear(semantic_features)
        attn_output, attn_weights = self.cross_attn(
            hidden_states=semantic_features,
            key_value_states=image_features,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=True,
        )
        return self.d_abstractor(attn_output)