import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, alphabet_size, tagset_size,
                 token_padding_idx, char_padding_idx,
                 char_level, token_level, pre_suf_level,
                 pre_vocab_size=None, suf_vocab_size=None, pre_padding_idx=None, suf_padding_idx=None,
                 pretrained_vecs=None):
        super(BiLSTM, self).__init__()
        # self.char_emb_size = 300
        # self.token_emb_size = 300
        # self.lstm_dim = 300
        # self.char_level = char_level
        # self.token_level = token_level
        # self.pre_suf_level = pre_suf_level

        # self.char_emb = nn.Embedding(alphabet_size, self.char_emb_size, padding_idx=char_padding_idx)
        # if self.pre_suf_level:
        #     self.pre_emb = nn.Embedding(pre_vocab_size, self.token_emb_size, padding_idx=pre_padding_idx)
        #     self.suf_emb = nn.Embedding(suf_vocab_size, self.token_emb_size, padding_idx=suf_padding_idx)
        # if pretrained_vecs is None:
        #     self.word_emb = nn.Embedding(vocab_size, self.token_emb_size, padding_idx=token_padding_idx)
        # else:
        #     self.word_emb = nn.Embedding.from_pretrained(pretrained_vecs, freeze=False, padding_idx=token_padding_idx)
        #
        # self.lstm_c = nn.LSTM(bidirectional=False, input_size=self.char_emb_size,
        #                       hidden_size=self.token_emb_size, num_layers=1, batch_first=True)
        # self.bilstm1 = nn.LSTM(bidirectional=True, input_size=self.token_emb_size,
        #                        hidden_size=self.lstm_dim, num_layers=1, batch_first=True)
        # self.bilstm2 = nn.LSTM(bidirectional=True, input_size=2*self.lstm_dim,
        #                        hidden_size=self.lstm_dim, num_layers=1, batch_first=True)

        # self.char_tokens_linear = nn.Linear(2*self.token_emb_size, self.token_emb_size)
        # self.linear = nn.Linear(2*self.lstm_dim, tagset_size)
        # self.dropout = nn.Dropout(p=0.3)

        # EDITS START
        self.token_emb_size = 300
        self.lstm_dim = 300
        self.word_emb = nn.Embedding.from_pretrained(pretrained_vecs, freeze=False, padding_idx=token_padding_idx)
        self.bilstm1 = nn.LSTM(bidirectional=True, input_size=self.token_emb_size,
                               hidden_size=self.lstm_dim, num_layers=1, batch_first=True)
        self.bilstm2 = nn.LSTM(bidirectional=True, input_size=2*self.lstm_dim,
                               hidden_size=self.lstm_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(2*self.lstm_dim, tagset_size)
        self.dropout = nn.Dropout(p=0.5)
        # EDITS END

    def forward(self, tokens_input_ids, pre_input_ids, suf_input_ids, char_input_ids, x_tokens_lens, x_chars_lens):
        # if self.char_level:
        #     flatten_x_chars_lens = [item for sublist in x_chars_lens for item in sublist]
        #     mask = torch.tensor(flatten_x_chars_lens, device=char_input_ids.device) > 0
        #     x_chars_lens = [l for l in flatten_x_chars_lens if l > 0]
        #
        #     embds = self.char_emb(char_input_ids)        # [batch, sequence_len, token_len, emb]
        #     batch, sequence_len, token_len, emb_size = embds.size()
        #
        #     embds = embds.view(-1, token_len, emb_size)     # [batch*sequence_len, token_len, emb]
        #     embds = embds[mask]
        #
        #     x_packed = pack_padded_sequence(embds, x_chars_lens, batch_first=True, enforce_sorted=False)
        #     _, (h_chars, c) = self.lstm_c(x_packed)
        #
        #     chars_reps = h_chars.new_zeros((batch, sequence_len, self.token_emb_size))
        #     total_l = 0
        #     for i, l in enumerate(x_tokens_lens):
        #         chars_reps[i, 0:l, :] = h_chars[:, total_l:total_l+l, :]
        #         total_l += l

        # if self.token_level:
        #     tokens_reps = self.word_emb(tokens_input_ids)  # [batch, sequence_len, emb]
        #     if self.pre_suf_level:
        #         tokens_reps += self.pre_emb(pre_input_ids)  # [batch, sequence_len, emb]
        #         tokens_reps += self.suf_emb(suf_input_ids)  # [batch, sequence_len, emb]
        # if self.char_level and self.token_level:
        #     reps = torch.cat([chars_reps, tokens_reps], dim=-1) # [batch, sequence_len, 2*token_emb]
        #     reps = self.char_tokens_linear(reps)                # [batch, sequence_len, 2*token_emb]
        # elif self.char_level:
        #     reps = chars_reps                                   # [batch, sequence_len, token_emb]
        # else:
        #     reps = tokens_reps                                  # [batch, sequence_len, token_emb]

        # EDITS START
        tokens_reps = self.word_emb(tokens_input_ids)  # [batch, sequence_len, emb]
        reps = tokens_reps                                  # [batch, sequence_len, token_emb]
        # EDITS END

        reps = self.dropout(reps)
        x_packed = pack_padded_sequence(reps, x_tokens_lens, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.bilstm1(x_packed)  # [batch, sequence_len, out_dim]
        lstm_out, _ = self.bilstm2(lstm_out)
        output_padded, output_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        output = self.linear(output_padded)
        return output