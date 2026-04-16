import torch
import torch.nn as nn

from .pixel_decoder import SimpleFPNPixelDecoder
from .position_encoding import PositionEmbeddingSine


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[index], dims[index + 1])
            for index in range(len(dims) - 1)
        )

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class Mask2FormerHead(nn.Module):
    def __init__(self, cfg, input_shapes):
        super().__init__()
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        mask_dim = cfg.MODEL.MASK_FORMER.MASK_DIM
        self.num_classes = cfg.MODEL.MASK_FORMER.NUM_CLASSES
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES

        self.pixel_decoder = SimpleFPNPixelDecoder(input_shapes, hidden_dim, mask_dim)
        self.input_proj = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) for _ in range(3)]
        )
        self.level_embed = nn.Embedding(3, hidden_dim)
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=cfg.MODEL.MASK_FORMER.NHEADS,
            dim_feedforward=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            dropout=cfg.MODEL.MASK_FORMER.DROPOUT,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cfg.MODEL.MASK_FORMER.DEC_LAYERS,
        )

        self.query_features = nn.Embedding(self.num_queries, hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, self.num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, features):
        mask_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        batch_size = mask_features.shape[0]

        memory = []
        for level, feature in enumerate(multi_scale_features):
            projected = self.input_proj[level](feature)
            positional = self.position_embedding(projected)
            projected = projected.flatten(2).permute(2, 0, 1)
            positional = positional.flatten(2).permute(2, 0, 1)
            level_bias = self.level_embed.weight[level].view(1, 1, -1)
            memory.append(projected + positional + level_bias)
        memory = torch.cat(memory, dim=0)

        query_features = self.query_features.weight.unsqueeze(1).repeat(1, batch_size, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        decoded = self.decoder(query_features + query_embed, memory)
        decoded = decoded.transpose(0, 1)

        pred_logits = self.class_embed(decoded)
        mask_embed = self.mask_embed(decoded)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return {"pred_logits": pred_logits, "pred_masks": pred_masks}