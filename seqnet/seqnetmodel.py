import torch
import torch.nn as nn
from seqnet.seq_net import RNN, SumAttention, HGN
from seqnet.bert_modules.bert import BERT
from seqnet.bert_modules.embedding import BERTEmbedding
from seqnet.bert_modules.embedding_fea import BERTEmbedding as BERTEmbedding_fea


class SEQNETMODEL(nn.Module):
    def __init__(self, config, num_items, num):
        super(SEQNETMODEL, self).__init__()
        self.config = config
        self.use_feature = config.use_feature
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
        self.gru_fea_embeddings0 = nn.Embedding(num[0], dim)
        if len(num) > 1:
            self.gru_fea_embeddings1 = nn.Embedding(num[1], dim)
            self.gru_fea_embeddings2 = nn.Embedding(num[2], dim)

        self.bert_items_embeddings = BERTEmbedding(vocab_size=num_items + 2, embed_size=dim,
                                                   max_len=config.bert_max_len, dropout=dropout)
        self.bert_fea_embeddings0 = BERTEmbedding_fea(vocab_size=num[0], embed_size=dim,
                                                   max_len=config.bert_max_len, dropout=dropout)
        if len(num)>1:
            self.bert_fea_embeddings1 = BERTEmbedding_fea(vocab_size=num[1], embed_size=dim,
                                                   max_len=config.bert_max_len, dropout=dropout)
            self.bert_fea_embeddings2 = BERTEmbedding_fea(vocab_size=num[2], embed_size=dim,
                                                   max_len=config.bert_max_len, dropout=dropout)

        # output
        self.mlp_gru = nn.Linear(2 * dim, num_items + 2)
        self.mlp_gru_fea0 = nn.Linear(2 * dim, num[0])
        if len(num)>1:
            self.mlp_gru_fea1 = nn.Linear(2 * dim, num[1])
            self.mlp_gru_fea2 = nn.Linear(2 * dim, num[2])

        self.mlp_bert = nn.Linear(dim, num_items + 2)
        self.mlp_bert_fea0 = nn.Linear(dim, num[0])
        if len(num)>1:
            self.mlp_bert_fea1 = nn.Linear(dim, num[1])
            self.mlp_bert_fea2 = nn.Linear(dim, num[2])


        # 线性门控单元
        self.m1 = nn.Linear(dim, dim)
        self.m2 = nn.Linear(dim, dim)

        # model
        self.rnn = RNN(rnn_type, input_size, hidden_size,
                       num_layers, dropout, bidirectional,
                       return_sequence)
        self.rnn0 = RNN(rnn_type, input_size, hidden_size,
                        num_layers, dropout, bidirectional,
                        return_sequence)
        if len(num) > 1:
            self.rnn1 = RNN(rnn_type, input_size, hidden_size,
                            num_layers, dropout, bidirectional,
                            return_sequence)
            self.rnn2 = RNN(rnn_type, input_size, hidden_size,
                            num_layers, dropout, bidirectional,
                            return_sequence)

        self.bert = BERT(config)
        self.bert0 = BERT(config)
        if len(num) > 1:
            self.bert1 = BERT(config)
            self.bert2 = BERT(config)

        self.attention = SumAttention(input_size, hidden_size)

    def attention_feature(self, item_embs, emb_0, emb_1, emb_2):
        emb_0 = self.attention(emb_0)
        emb_1 = self.attention(emb_1)
        emb_2 = self.attention(emb_2)

        alpha_0 = torch.mean(item_embs * emb_0)
        alpha_1 = torch.mean(item_embs * emb_1)
        alpha_2 = torch.mean(item_embs * emb_2)

        alpha_0 = torch.exp(alpha_0)
        alpha_1 = torch.exp(alpha_1)
        alpha_2 = torch.exp(alpha_2)

        alpha_0 = alpha_0 / (alpha_0 + alpha_1 + alpha_2)
        alpha_1 = alpha_1 / (alpha_0 + alpha_1 + alpha_2)
        alpha_2 = alpha_2 / (alpha_0 + alpha_1 + alpha_2)

        return alpha_0*emb_0 + alpha_1*emb_1 + alpha_2*emb_2
                     # 1024 11    1024 11 4
    def forward(self, item_seq, batch_feature_0=None, batch_feature_1=None, batch_feature_2=None):
        if self.config.model == "gru":
            if batch_feature_1 is None:
                item_embs = self.gru_items_embeddings(item_seq) #1024 11 256
                fea_emb_0 = self.gru_fea_embeddings0(batch_feature_0) #1024 11 4 256
                emb_feature = self.attention(fea_emb_0) # 1024 11 256
                item_embs = 0.5 * item_embs + 0.5 * emb_feature
                #item_embs = item_embs*torch.sigmoid(self.m1(item_embs)+self.m2(emb_feature))
                output = self.rnn(item_embs)
                output0 = self.rnn0(emb_feature)
                #output = torch.cat((output, output0), 2)
                return self.mlp_gru(output), self.mlp_gru_fea0(output0)
                # 1024 512
            if batch_feature_1 is not None:
                item_embs = self.gru_items_embeddings(item_seq)
                fea_emb_0 = self.gru_fea_embeddings0(batch_feature_0)
                emb_feature_0 = self.attention(fea_emb_0)
                fea_emb_1 = self.gru_fea_embeddings1(batch_feature_1)
                emb_feature_1 = self.attention(fea_emb_1)
                fea_emb_2 = self.gru_fea_embeddings2(batch_feature_2)
                emb_feature_2 = self.attention(fea_emb_2)

                emb_feature = self.attention_feature(item_embs, emb_feature_0, emb_feature_1, emb_feature_2)
                item_embs = 0.5 * item_embs + 0.5 * emb_feature

                output = self.rnn(item_embs)
                output0 = self.rnn0(emb_feature_0)
                output1 = self.rnn1(emb_feature_1)
                output2 = self.rnn2(emb_feature_2)
                #output = torch.cat((output, output0), 2)

                return self.mlp_gru(output), self.mlp_gru_fea0(output0)\
                    , self.mlp_gru_fea1(output1), self.mlp_gru_fea2(output2)

        elif self.config.model == "bert":
            if batch_feature_1 is None:
                item_embs = self.bert_items_embeddings(item_seq) # 1024 8 256
                item_mask = (item_seq > 0).unsqueeze(1).repeat(1, item_seq.size(1), 1).unsqueeze(1) # 1024 1 8 8
                fea_emb_0 = self.bert_fea_embeddings0(batch_feature_0) # 1024 8 4 256
                emb_feature = self.attention(fea_emb_0) # 1024 8 256
                item_embs = 0.5 * item_embs + 0.5 * emb_feature
                # item_embs = item_embs*torch.sigmoid(self.m1(item_embs)+self.m2(emb_feature)) # 1024 8 256
                output = self.bert(item_embs, item_mask) # 1024 8 256
                output0 = self.bert0(emb_feature, item_mask) # 1024 8 256
                # output = torch.cat((output, output0), 2) # 1024 8 512
                return self.mlp_bert(output), self.mlp_bert_fea0(output0)

            if batch_feature_1 is not None:
                item_embs = self.bert_items_embeddings(item_seq)
                item_mask = (item_seq > 0).unsqueeze(1).repeat(1, item_seq.size(1), 1).unsqueeze(1)
                fea_emb_0 = self.bert_fea_embeddings0(batch_feature_0)
                emb_feature_0 = self.attention(fea_emb_0)
                fea_emb_1 = self.bert_fea_embeddings1(batch_feature_1)
                emb_feature_1 = self.attention(fea_emb_1)
                fea_emb_2 = self.bert_fea_embeddings2(batch_feature_2)
                emb_feature_2 = self.attention(fea_emb_2)

                emb_feature = self.attention_feature(item_embs, emb_feature_0, emb_feature_1, emb_feature_2)
                item_embs = 0.5 * item_embs + 0.5 * emb_feature
                #item_embs = item_embs*torch.sigmoid(self.m1(item_embs)+self.m2(emb_feature))
                output = self.bert(item_embs, item_mask)
                output0 = self.bert0(emb_feature_0, item_mask)
                output1 = self.bert1(emb_feature_1, item_mask)
                output2 = self.bert2(emb_feature_2, item_mask)

                #output = torch.cat((output, output0), 2)
                return self.mlp_bert(output), self.mlp_bert_fea0(output0)\
                    , self.mlp_bert_fea1(output1), self.mlp_bert_fea2(output2)

        elif self.config.model == "hgn":
            #output = self.hgn(item_seq, user_ids, items_to_predict)
            #return output
            return



