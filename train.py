import argparse
import random
import torch
import numpy as np
import pickle
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# from data import PAD, TagDataset, load_pretrained_embeds
from data import load_glove_vectors
from models import BiLSTM

def train():
    pass

if __name__ == '__main__':
    random.seed(6)
    np.random.seed(6)
    torch.manual_seed(6)

    emb_file = 'data/glove.42B.300d.txt'
    emb, vocab = load_glove_vectors('data/glove.42B.300d.txt')
    with open('data/glove.42B.300d.pkl', 'wb') as f:
        pickle.dump((emb, vocab), f)
    # with open('data/glove.42B.300d.pkl', 'rb') as f:
    #     emb, vocab = pickle.load(f)

    print(emb.shape)

    train_dataset = SNLIDataset(dataset=load_dataset('snli', split='train', vocab=vocab))
    print(train_dataset[0])