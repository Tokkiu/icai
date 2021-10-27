import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size,
                 num_layers, dropout, bidirectional,
                 return_sequence):
        """init"""
        super(RNN, self).__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = True
        self.rnn = None
        self.return_sequence = return_sequence

        if self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=self.num_layers,
                batch_first=self.batch_first,
                dropout=dropout,
                bidirectional=self.bidirectional)
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=self.num_layers,
                batch_first=self.batch_first,
                dropout=dropout,
                bidirectional=self.bidirectional)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=self.num_layers,
                batch_first=self.batch_first,
                dropout=dropout,
                bidirectional=self.bidirectional)
        else:
            raise RuntimeError("Unknown RNN Type: " + self.rnn_type)

    def forward(self, inputs):
        """ rnn outputs"""
        self.rnn.flatten_parameters()  # self.rnn是我所使用的RNN
        sequence_output, hn = self.rnn(inputs)
        return sequence_output[:,-1,:]


class SumAttention(nn.Module):
    def __init__(self, input_dimension, attention_dimension):
        super(SumAttention, self).__init__()
        self.linear_u = nn.Linear(input_dimension, attention_dimension, bias=True)
        self.linear_v = nn.Linear(attention_dimension, 1, bias=False)

    def forward(self, inputs):
        if inputs.size(1) == 1:
            return inputs.squeeze()
        u = torch.tanh(self.linear_u(inputs))
        v = self.linear_v(u)

        #alpha = nn.functional.softmax(v, 2).squeeze(3).unsqueeze(2)
        if len(nn.functional.softmax(v, 2).size())>3:
            alpha = nn.functional.softmax(v, 2).squeeze(3).unsqueeze(2)
        else:
            alpha = nn.functional.softmax(v, 2).unsqueeze(2)

        if len(alpha.size()) == 3:
            alpha = alpha.unsqueeze(3)

        return torch.matmul(alpha, inputs).squeeze(2)


class HGN(nn.Module):
    def __init__(self, num_users, num_items, model_args):
        super(HGN, self).__init__()

        self.args = model_args

        # init args
        L = self.args.L+1
        dims = self.args.dim

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        self.feature_gate_item = nn.Linear(dims, dims)
        self.feature_gate_user = nn.Linear(dims, dims)

        self.end_mlp = nn.Linear(dims, dims)
        self.helper = nn.Linear(L, 256)

        self.instance_gate_item = torch.nn.Parameter(torch.zeros(dims, 1).type(torch.FloatTensor), requires_grad=True)
        self.instance_gate_user = torch.nn.Parameter(torch.zeros(dims, L).type(torch.FloatTensor), requires_grad=True)

        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)

        self.W2 = nn.Embedding(num_items, dims, padding_idx=0)
        self.b2 = nn.Embedding(num_items, 1, padding_idx=0)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self, item_seq, user_ids, items_to_predict,
                batch_feature_0=None, batch_feature_1=None, batch_feature_2=None,
                features_to_predict=None):
        item_embs = self.item_embeddings(item_seq)
        user_emb = self.user_embeddings(user_ids)

        # feature gating
        gate = torch.sigmoid(self.feature_gate_item(item_embs) + self.feature_gate_user(user_emb).unsqueeze(1))
        gated_item = item_embs * gate

        # instance gating
        instance_score = torch.sigmoid(torch.matmul(gated_item, self.instance_gate_item.unsqueeze(0)).squeeze() + user_emb.mm(self.instance_gate_user))
        union_out = gated_item * instance_score.unsqueeze(2)
        union_out = torch.sum(union_out, dim=1)
        union_out = union_out / torch.sum(instance_score, dim=1).unsqueeze(1)

        w2 = self.W2(items_to_predict)
        b2 = self.b2(items_to_predict)

        res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()

        # union-level
        res += torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

        # item-item product
        rel_score = item_embs.bmm(w2.permute(0, 2, 1))
        rel_score = torch.sum(rel_score, dim=1)
        res += rel_score

        return res
