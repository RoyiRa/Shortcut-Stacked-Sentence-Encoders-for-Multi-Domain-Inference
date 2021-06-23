import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class StackBiLSTMMaxout(nn.Module):
    def __init__(self, emb, padding_idx):
        super(StackBiLSTMMaxout, self).__init__()

        d = emb.shape[1]
        h = [512, 1024, 2048]
        mlp_d = 1600

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=False, padding_idx=padding_idx)

        self.bilstm0 = nn.LSTM(input_size=d, hidden_size=h[0],
                            num_layers=1, bidirectional=True)

        self.bilstm1 = nn.LSTM(input_size=(d + 2*h[0]), hidden_size=h[1],
                              num_layers=1, bidirectional=True)

        self.bilstm2 = nn.LSTM(input_size=(d + 2*h[0] + 2*h[1]), hidden_size=h[2],
                              num_layers=1, bidirectional=True)

        self.mlp_1 = nn.Linear(h[2]*2*4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, 3)

        p = 0.1
        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(p),
                                          self.mlp_2, nn.ReLU(), nn.Dropout(p),
                                          self.sm])

    def max_along_time(self, inputs, lengths):
        """
        :param inputs: [B * T * D]
        :param lengths:  [B]
        :return: [B * D] max_along_time
        """
        b_seq_max_list = []
        for i, l in enumerate(lengths):
            seq_i = inputs[i, :l, :]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)

    def forward(self, premise_ids, hypothesis_ids, premise_lens, hypothesis_lens):
        pre_reps = self.embedding(premise_ids)          # [batch, max_sent_len, d]
        hyp_reps = self.embedding(hypothesis_ids)       # [batch, max_sent_len, d]

        pre_packed = pack_padded_sequence(pre_reps, premise_lens, batch_first=True, enforce_sorted=False)
        hyp_packed = pack_padded_sequence(hyp_reps, hypothesis_lens, batch_first=True, enforce_sorted=False)
        pre_lstm0_out, _ = self.bilstm0(pre_packed)
        hyp_lstm0_out, _ = self.bilstm0(hyp_packed)
        pre_lstm0_out, _ = pad_packed_sequence(pre_lstm0_out, batch_first=True)
        hyp_lstm0_out, _ = pad_packed_sequence(hyp_lstm0_out, batch_first=True)

        pre_reps = torch.cat([pre_lstm0_out, pre_reps], dim=-1)
        hyp_reps = torch.cat([hyp_lstm0_out, hyp_reps], dim=-1)

        pre_packed = pack_padded_sequence(pre_reps, premise_lens, batch_first=True, enforce_sorted=False)
        hyp_packed = pack_padded_sequence(hyp_reps, hypothesis_lens, batch_first=True, enforce_sorted=False)
        pre_lstm1_out, _ = self.bilstm1(pre_packed)
        hyp_lstm1_out, _ = self.bilstm1(hyp_packed)
        pre_lstm1_out, _ = pad_packed_sequence(pre_lstm1_out, batch_first=True)
        hyp_lstm1_out, _ = pad_packed_sequence(hyp_lstm1_out, batch_first=True)

        pre_reps = torch.cat([pre_lstm1_out, pre_reps], dim=-1)
        hyp_reps = torch.cat([hyp_lstm1_out, hyp_reps], dim=-1)

        pre_packed = pack_padded_sequence(pre_reps, premise_lens, batch_first=True, enforce_sorted=False)
        hyp_packed = pack_padded_sequence(hyp_reps, hypothesis_lens, batch_first=True, enforce_sorted=False)
        pre_lstm2_out, _ = self.bilstm2(pre_packed)
        hyp_lstm2_out, _ = self.bilstm2(hyp_packed)
        pre_lstm2_out, _ = pad_packed_sequence(pre_lstm2_out, batch_first=True)
        hyp_lstm2_out, _ = pad_packed_sequence(hyp_lstm2_out, batch_first=True)

        fin_pre = self.max_along_time(pre_lstm2_out, premise_lens)
        fin_hyp = self.max_along_time(hyp_lstm2_out, hypothesis_lens)

        features = torch.cat([fin_pre, fin_hyp, torch.abs(fin_pre - fin_hyp), fin_pre * fin_hyp], dim=1)

        out = self.classifier(features)

        return out





