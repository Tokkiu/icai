import torch
import torch.nn as nn
from seqnet.seq_net import RNN, HGN
from seqnet.bert_modules.bert import BERT
from seqnet.bert_modules.embedding import BERTEmbedding


class SEQMODEL(nn.Module):
    def __init__(self, config, num_items, num_user):
        super(SEQMODEL, self).__init__()
        self.use_feature = config.use_feature
        self.config = config
        rnn_type = config.rnn_type
        input_size = config.dim
        hidden_size = config.dim
        num_layers = config.num_layers
        dropout = config.dropout
        bidirectional = config.bidirectional
        return_sequence = config.return_sequence
        dim = config.dim

        # input
        self.gru_items_embeddings = nn.Embedding(num_items + 2, dim)
        self.bert_items_embeddings = BERTEmbedding(vocab_size=num_items + 2, embed_size=dim, max_len=config.bert_max_len, dropout=dropout)

        # output
        self.mlp_gru = nn.Linear(2 * dim, num_items + 2)
        self.mlp_bert = nn.Linear(dim, num_items + 2)

        # model
        self.rnn = RNN(rnn_type, input_size, hidden_size,
                       num_layers, dropout, bidirectional,
                       return_sequence)
        self.bert = BERT(config)
        self.hgn = HGN(num_user, num_items+2, config)


    def forward(self, item_seq, user_ids=None, items_to_predict=None):
        if self.config.model == "gru":
            item_embs = self.gru_items_embeddings(item_seq)
            output = self.rnn(item_embs)
            return self.mlp_gru(output)
        elif self.config.model == "bert":
            item_embs = self.bert_items_embeddings(item_seq)
            mask = (item_seq > 0).unsqueeze(1).repeat(1, item_seq.size(1), 1).unsqueeze(1)
            output = self.bert(item_embs, mask)
            return self.mlp_bert(output)
        elif self.config.model == "hgn":
            output = self.hgn(item_seq, user_ids, items_to_predict)
            return output





