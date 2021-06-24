import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class StackBiLSTMMaxout(nn.Module):
    def __init__(self, emb, padding_idx):
        super(StackBiLSTMMaxout, self).__init__()

        d = emb.shape[1]
        h = [300, 300, 300]
        mlp_d = 800

        # v_size = 10
        # self.Embd = nn.Embedding(v_size, d)

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(emb), freeze=False, padding_idx=padding_idx)

        self.bilstm0 = nn.LSTM(input_size=d, hidden_size=h[0], num_layers=1, bidirectional=True)

        self.bilstm1 = nn.LSTM(input_size=(d + 2*h[0]), hidden_size=h[1], num_layers=1, bidirectional=True)

        self.bilstm2 = nn.LSTM(input_size=(d + 2*h[0]), hidden_size=h[2], num_layers=1, bidirectional=True)

        self.mlp_1 = nn.Linear(h[2]*2*4, mlp_d)
        self.sm = nn.Linear(mlp_d, 3)

        p = 0.1
        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(p),
                                          self.sm])

    def display(self):
        total_params = 0
        emb_size = 0
        for n, p in self.named_parameters():
            print(f'{n}: {p.size()}')
            layer_p = 1
            for s in p.size():
                layer_p *= s
            if 'embedding' in n:
                emb_size = layer_p
            else:
                total_params += layer_p
        print(f'Total Model Parameters size wo embedding : {(total_params / 1000000):.1f}M')
        print(f'Total Model Parameters size w embedding : {((total_params + emb_size) / 1000000):.1f}M')

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

    def forward_lstm(self, lstm, input, input_len):
        input = torch.cat(input, dim=-1)

        packed = pack_padded_sequence(input, input_len, batch_first=True, enforce_sorted=False)
        lstm_out, _ = lstm(packed)
        out_padded, _ = pad_packed_sequence(lstm_out, batch_first=True)

        return out_padded

    def forward(self, premise_ids, hypothesis_ids, premise_lens, hypothesis_lens):
        pre_reps = self.embedding(premise_ids)          # [batch, max_sent_len, d]
        hyp_reps = self.embedding(hypothesis_ids)       # [batch, max_sent_len, d]

        pre_lstm0_out = self.forward_lstm(self.bilstm0, [pre_reps], premise_lens)
        hyp_lstm0_out = self.forward_lstm(self.bilstm0, [hyp_reps], hypothesis_lens)

        pre_lstm1_out = self.forward_lstm(self.bilstm1, [pre_lstm0_out, pre_reps], premise_lens)
        hyp_lstm1_out = self.forward_lstm(self.bilstm1, [hyp_lstm0_out, hyp_reps], hypothesis_lens)

        pre_lstm2_out = self.forward_lstm(self.bilstm2, [pre_lstm1_out + pre_lstm0_out, pre_reps], premise_lens)
        hyp_lstm2_out = self.forward_lstm(self.bilstm2, [hyp_lstm1_out + hyp_lstm0_out, hyp_reps], hypothesis_lens)

        fin_pre = self.max_along_time(pre_lstm2_out, premise_lens)
        fin_hyp = self.max_along_time(hyp_lstm2_out, hypothesis_lens)

        features = torch.cat([fin_pre, fin_hyp, torch.abs(fin_pre - fin_hyp), fin_pre * fin_hyp], dim=1)

        out = self.classifier(features)

        return out





