import torch
import torch.nn as nn

import esm

from modeling.utils import FFNLayer

class ESMBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        #getattr()，动态调用。getattr(object, name[, default]) -> value，name动态访问属性,以便更加模块化。
        self.backbone, self.alphabet = getattr(esm.pretrained, args.backbone)()
        self.num_layers = len(self.backbone.layers)
        self.hdim = self.backbone.lm_head.dense.weight.shape[1]

        if args.freeze_at > 0:
            self.backbone.embed_tokens.requires_grad_(False)
            for i, layer in enumerate(self.backbone.layers):
                if i < args.freeze_at:
                    layer.requires_grad_(False)

        ## prepare feature extractor
        self.ln = nn.LayerNorm(self.hdim)
        self.bb_adapter = FFNLayer(self.hdim, self.hdim)

        ## prepare penultimate aa_type embeddings
        self.aa_expand = args.aa_expand
        if self.aa_expand == 'backbone':
            self.aa_embed = self.backbone.lm_head.weight.requires_grad_(True)
            self.esm_to_our_aatype = [self.alphabet.get_idx(aa) for aa in one_letters]
            self.ln_head = nn.LayerNorm(self.get_aa_embed_dim())

        ## avoid distributed issues
        self.backbone.lm_head.requires_grad_(False)
        self.backbone.contact_head.requires_grad_(False)

    def get_aa_embed_dim(self):
        return self.aa_embed.shape[1]

    def get_aa_embed(self):
        return self.ln_head(self.aa_embed[self.esm_to_our_aatype])

    def get_alphabet(self):
        return 'esm', self.alphabet

    def forward(self, x, batch):
        x = self.backbone(x, repr_layers=[self.num_layers])['representations'][self.num_layers]
        if len(x.shape) == 4:  # remove MSAs
            x = x[:,0]
        x = self.bb_adapter(self.ln(x))
        x = x[:, 1:-1]  # remove SOS and EOS tokens
        ret = {'bb_feat': x}
        if self.aa_expand == 'backbone':
            ret['aa_embed'] = self.get_aa_embed()
        return ret