import os

os.environ['HF_HOME'] = 'D:/hugging face'
os.environ['TRANSFORMERS_CACHE'] = 'D:/hugging face/cache'

import torch
import torch.nn as nn
import esm

class FFNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFNLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class ESMBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.backbone, self.alphabet = getattr(esm.pretrained, args.backbone)()
        # load from local path D:/hugging face/esm1b_t33_650M_UR50S
        self.backbone, self.alphabet = esm.pretrained.load_model_and_alphabet_local('D:/hugging face/esm1b_t33_650M_UR50S.pt')

        self.num_layers = len(self.backbone.layers)
        self.hdim = self.backbone.lm_head.dense.weight.shape[1]

        if args.freeze_at > 0:
            self.backbone.embed_tokens.requires_grad_(False)
            for i, layer in enumerate(self.backbone.layers):
                if i < args.freeze_at:
                    layer.requires_grad_(False)

        self.ln = nn.LayerNorm(self.hdim)
        self.bb_adapter = FFNLayer(self.hdim, self.hdim)

        self.aa_expand = args.aa_expand
        if self.aa_expand == 'backbone':
            self.aa_embed = self.backbone.lm_head.weight.requires_grad_(True)
            self.esm_to_our_aatype = [self.alphabet.get_idx(aa) for aa in one_letters]
            self.ln_head = nn.LayerNorm(self.get_aa_embed_dim())

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
        if len(x.shape) == 4:
            x = x[:,0]
        x = self.bb_adapter(self.ln(x))
        x = x[:, 1:-1]
        ret = {'bb_feat': x}
        if self.aa_expand == 'backbone':
            ret['aa_embed'] = self.get_aa_embed()
        return ret
