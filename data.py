from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np

PAD = '<PAD>'
UNK = '<UNK>'


# Data Fields
# premise: a string used to determine the truthfulness of the hypothesis
# hypothesis: a string that may be true, false, or whose truth conditions may not be knowable when compared to the premise
# label: an integer whose value may be either
#           0, indicating that the hypothesis entails the premise,
#           1, indicating that the premise and hypothesis neither entail nor contradict each other
#           2, indicating that the hypothesis contradicts the premise.
# Dataset instances which don't have any gold label are marked with -1 label. Make sure you filter them before starting the training using datasets.Dataset.filter.


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

    return np.array(W), vocab


class SNLIDataset(Dataset):
    PREMISE = 'premise'
    HYPOTHESIS = 'hypothesis'
    LABEL = 'label'

    def __init__(self, dataset, vocab):
        self.dataset = dataset
        self.vocab = vocab

        self.examples = [self._tensorize_example(ex) for ex in dataset]

    def _tensorize_example(self, example):
        premise, hypothesis, label = example[self.PREMISE], example[self.HYPOTHESIS], example[self.LABEL]

        premise_ids = [self.vocab[t] if t in self.vocab else self.vocab[UNK] for t in premise.split()]
        hypothesis_ids = [self.vocab[t] if t in self.vocab else self.vocab[UNK] for t in hypothesis.split()]

        premise_ids = torch.tensor(premise_ids, dtype=torch.long)
        hypothesis_ids = torch.tensor(hypothesis_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return premise_ids, hypothesis_ids, label

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]