import torch
from torch import nn
from esm.modules import ESM1bLayerNorm, TransformerLayer


class PWRules(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.embed_dim = 1280
        self.attention_heads = 20
        self.num_layers = 1
        self.output_dim = output_dim

        self.embedding_layer = nn.Embedding(num_embeddings=1, embedding_dim=1280)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.output_dim),
        )

    def forward(self, x, repr_layers=[], need_head_weights=False):
        cls_input = torch.zeros((x.shape[0],), dtype=torch.long).to(x.device)
        cls_embedding = self.embedding_layer(cls_input).unsqueeze(1)

        x = torch.cat((cls_embedding, x), dim=1).detach()

        padding_mask = (x.abs().sum(dim=-1) == 0)

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.mlp(x[:, 0, :])

        result = {"predict": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
        return result
