from torch import nn as nn

from seqnet.bert_modules.embedding import BERTEmbedding
from seqnet.bert_modules.transformer import TransformerBlock
from seqnet.bert_modules.util import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()

        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads

        hidden = args.bert_hidden_units
        self.hidden = hidden
        dropout = args.bert_dropout

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, mask):
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass
