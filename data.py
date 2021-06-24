import torchtext
from torch.utils.data import Dataset
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


def load_glove_vectors(file, cache_dir):
    glove = torchtext.vocab.Vectors(name=file, cache=cache_dir)

    vectors = glove.vectors
    vocab = glove.stoi

    vocab[UNK] = len(vocab)
    unk_vec = vectors.mean(dim=0, keepdim=True)
    vectors = torch.cat([vectors, unk_vec], dim=0)

    vocab[PAD] = len(vocab)
    pad_vec = torch.zeros((1, vectors.shape[1]))
    vectors = torch.cat([vectors, pad_vec], dim=0)

    return vectors, vocab


class SNLIDataset(Dataset):

    def __init__(self, dataset, vocab, tokenizer, lower):
        self.dataset = dataset
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.lower = lower

        self.examples = [self._tensorize_example(ex) for ex in dataset if ex[LABEL] != -1]

    def _tensorize_example(self, example):
        premise, hypothesis, label = example[PREMISE], example[HYPOTHESIS], example[LABEL]

        if self.lower:
            pre_tokens = [t.text.lower() for t in self.tokenizer(premise)]
            hyp_tokens = [t.text.lower() for t in self.tokenizer(hypothesis)]
        else:
            pre_tokens = [t.text for t in self.tokenizer(premise)]
            hyp_tokens = [t.text for t in self.tokenizer(hypothesis)]

        premise_ids = [self.vocab[t] if t in self.vocab else self.vocab[UNK] for t in pre_tokens]
        hypothesis_ids = [self.vocab[t] if t in self.vocab else self.vocab[UNK] for t in hyp_tokens]

        premise_ids = torch.tensor(premise_ids, dtype=torch.long)
        hypothesis_ids = torch.tensor(hypothesis_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return {PREMISE: premise_ids, HYPOTHESIS: hypothesis_ids, LABEL: label}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]