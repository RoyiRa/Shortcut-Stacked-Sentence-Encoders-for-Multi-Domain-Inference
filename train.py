import argparse
import random
import torch
import numpy as np
import pickle
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

# from data import PAD, TagDataset, load_pretrained_embeds
from data import load_glove_vectors, SNLIDataset, PAD, PREMISE, HYPOTHESIS, LABEL
from models import BiLSTM


def pad_collate(batch, token_pad):
    xx_pre = [item[PREMISE] for item in batch]
    xx_hyp = [item[HYPOTHESIS] for item in batch]
    yy = [item[LABEL] for item in batch]

    x_pre_lens = [len(x) for x in xx_pre]
    x_hyp_lens = [len(x) for x in xx_hyp]

    xx_pre_pad = pad_sequence(xx_pre, batch_first=True, padding_value=token_pad)
    xx_hyp_pad = pad_sequence(xx_hyp, batch_first=True, padding_value=token_pad)

    return xx_pre_pad, xx_hyp_pad, x_pre_lens, x_hyp_lens, yy

def train():
    pass

if __name__ == '__main__':
    random.seed(6)
    np.random.seed(6)
    torch.manual_seed(6)

    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.tokenizer

    # emb_file = 'data/glove.42B.300d.txt'
    # emb, vocab = load_glove_vectors('data/glove.42B.300d.txt')
    # with open('data/glove.42B.300d.pkl', 'wb') as f:
    #     pickle.dump((emb, vocab), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/glove.42B.300d.pkl', 'rb') as f:
        emb, vocab = pickle.load(f)

    print(emb.shape)

    train_dataset = SNLIDataset(dataset=load_dataset('snli', split='test'), vocab=vocab, tokenizer=tokenizer)
    print(train_dataset.dataset[0])
    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              collate_fn=lambda b: pad_collate(b, vocab[PAD]))

    for x in train_loader:
        print(x)