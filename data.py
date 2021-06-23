from torch.utils.data import Dataset
import numpy as np
import torch


PAD = '<PAD>'
UNK = '<UNK>'
PREMISE = 'premise'
HYPOTHESIS = 'hypothesis'
LABEL = 'label'


# Data Fields
# premise: a string used to determine the truthfulness of the hypothesis
# hypothesis: a string that may be true, false, or whose truth conditions may not be knowable when compared to the premise
# label: an integer whose value may be either
#           0, indicating that the hypothesis entails the premise,
#           1, indicating that the premise and hypothesis neither entail nor contradict each other
#           2, indicating that the hypothesis contradicts the premise.
# Dataset instances which don't have any gold label are marked with -1 label.
# Make sure you filter them before starting the training using datasets.Dataset.filter.


def load_glove_vectors(file):
    W = []
    vocab = {}
    with open(file, 'r', encoding='utf8') as f:
        for row in f:
            row = row.split()
            vocab[row[0]] = len(vocab)

            v = np.array(row[1:], dtype=np.float)
            W.append(v)
    d = W[0].shape[0]

    vocab[UNK] = len(vocab)
    W.append(np.mean(W, axis=0))

    vocab[PAD] = len(vocab)
    W.append(np.zeros(d))

    return np.array(W, dtype=np.float32), vocab


class SNLIDataset(Dataset):


    def __init__(self, dataset, vocab, tokenizer):
        self.dataset = dataset
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.examples = [self._tensorize_example(ex) for ex in dataset if ex[LABEL] != -1]

    def _tensorize_example(self, example):
        premise, hypothesis, label = example[PREMISE], example[HYPOTHESIS], example[LABEL]


        tokens = [t.text.lower() for t in self.tokenizer(premise)]
        premise_ids = [self.vocab[t.lower()] if t.lower() in self.vocab else self.vocab[UNK] for t in tokens]

        tokens = [t.text.lower() for t in self.tokenizer(hypothesis)]
        hypothesis_ids = [self.vocab[t.lower()] if t.lower() in self.vocab else self.vocab[UNK] for t in tokens]

        premise_ids = torch.tensor(premise_ids, dtype=torch.long)
        hypothesis_ids = torch.tensor(hypothesis_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return {PREMISE: premise_ids, HYPOTHESIS: hypothesis_ids, LABEL: label}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]