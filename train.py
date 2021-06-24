import random
import torch
import numpy as np
import pickle
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset
from spacy.lang.en import English
from transformers import get_linear_schedule_with_warmup
from data import load_glove_vectors, SNLIDataset, PAD, PREMISE, HYPOTHESIS, LABEL
from model import ResStackBiLSTMMaxout
from tqdm import trange
from tqdm import tqdm


def pad_collate(batch, token_pad):
    xx_pre = [item[PREMISE] for item in batch]
    xx_hyp = [item[HYPOTHESIS] for item in batch]
    yy = torch.stack([item[LABEL] for item in batch], dim=0)

    x_pre_lens = [len(x) for x in xx_pre]
    x_hyp_lens = [len(x) for x in xx_hyp]

    xx_pre_pad = pad_sequence(xx_pre, batch_first=True, padding_value=token_pad)
    xx_hyp_pad = pad_sequence(xx_hyp, batch_first=True, padding_value=token_pad)

    return xx_pre_pad, xx_hyp_pad, x_pre_lens, x_hyp_lens, yy


def train(model, train_loader, dev_loader, device, batch, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=2e-4)
    t_total = (len(train_loader.dataset) // batch) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    train_loss = 0
    seen_examples = 0
    best_acc = 0.0

    accuracies = []
    losses = []
    steps = []
    state_dict = None

    eval_freq = 1024 * 50

    train_iterator = trange(0, epochs, desc="Epoch", position=0, leave=True)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration", position=0, leave=True)
        for step, (xx_pre_pad, xx_hyp_pad, x_pre_lens, x_hyp_lens, yy) in enumerate(epoch_iterator):
            model.train()
            model.zero_grad()

            premise_ids = xx_pre_pad.to(device)
            hypothesis_ids = xx_hyp_pad.to(device)
            yy = yy.to(device)
            premise_lens, hypothesis_lens = x_pre_lens, x_hyp_lens

            logits = model(premise_ids, hypothesis_ids, premise_lens, hypothesis_lens)

            loss = criterion(logits, yy)
            loss.backward()

            optimizer.step()
            scheduler.step()

            seen_examples += logits.shape[0]
            train_loss += loss.item()

            if seen_examples % eval_freq == 0:
                print()
                print(f'Train loss: {(train_loss / eval_freq):.8f}')

                acc = predict(model, dev_loader, device)
                if acc > best_acc:
                    best_acc = acc
                    state_dict = model.state_dict()
                print(f'Epoch: {epoch} dev acc: {acc:.8f}')
                print(f'Epoch: {epoch} best dev acc: {best_acc:.8f}')

                steps.append(seen_examples)
                accuracies.append(acc)
                losses.append(train_loss)
                train_loss = 0

    print(f'Train loss: {(train_loss / eval_freq):.8f}')
    acc = predict(model, dev_loader, device)
    if acc > best_acc:
        best_acc = acc
        state_dict = model.state_dict()
    print(f'Epoch: {epoch} dev acc: {acc:.8f}')
    print(f'Epoch: {epoch} best dev acc: {best_acc:.8f}')

    steps.append(seen_examples)
    accuracies.append(acc)
    losses.append(train_loss)

    return best_acc, accuracies, losses, steps, state_dict


def predict(model, loader, device):
    preds = []
    y = []

    model.eval()
    with torch.no_grad():
        for step, (xx_pre_pad, xx_hyp_pad, x_pre_lens, x_hyp_lens, yy) in enumerate(loader):
            premise_ids = xx_pre_pad.to(device)
            hypothesis_ids = xx_hyp_pad.to(device)
            yy = yy.to(device)
            premise_lens, hypothesis_lens = x_pre_lens, x_hyp_lens

            logits = model(premise_ids, hypothesis_ids, premise_lens, hypothesis_lens)

            preds += logits.softmax(dim=1).argmax(dim=1).tolist()
            y += yy.tolist()

    return np.sum(np.array(preds) == np.array(y)) / len(y)


if __name__ == '__main__':
    random.seed(6)
    np.random.seed(6)
    torch.manual_seed(6)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emb_file = 'data/glove.6B.300d.txt'
    batch = 32
    epochs = 3

    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.tokenizer

    emb, vocab = load_glove_vectors(emb_file)
    with open(emb_file.split('.')[0] + '.pkl', 'wb') as f:
        pickle.dump((emb, vocab), f, protocol=pickle.HIGHEST_PROTOCOL)

    ### use it if you have the emb in pickle
    # with open(emb_file.split('.')[0] + '.pkl', 'rb') as f:
    #     emb, vocab = pickle.load(f)

    model = ResStackBiLSTMMaxout(emb=emb, padding_idx=vocab[PAD])
    model.display()

    train_dataset = SNLIDataset(dataset=load_dataset('snli', split='train'), vocab=vocab, tokenizer=tokenizer)
    dev_dataset = SNLIDataset(dataset=load_dataset('snli', split='validation'), vocab=vocab, tokenizer=tokenizer)
    test_dataset = SNLIDataset(dataset=load_dataset('snli', split='test'), vocab=vocab, tokenizer=tokenizer)

    print(f'Pre-train embedding shape: {emb.shape}, vocab size:{len(vocab)}')
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Dev dataset size: {len(dev_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, collate_fn=lambda b: pad_collate(b, vocab[PAD]))
    dev_loader = DataLoader(dev_dataset, batch_size=batch, shuffle=False, collate_fn=lambda b: pad_collate(b, vocab[PAD]))
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, collate_fn=lambda b: pad_collate(b, vocab[PAD]))

    model = ResStackBiLSTMMaxout(emb=emb, padding_idx=vocab[PAD])
    model.to(device)

    best_acc, accuracies, losses, steps, state_dict = train(model, train_loader, dev_loader, device, batch, epochs)

    print(f'steps = {steps}')
    print(f'accuracies = {accuracies}')
    print(f'accuracies = {losses}')
    print(f'best_acc_dev: {best_acc}')

    model_path = 'data/model.pt'
    print(f'Saving model and train dataset to: {model_path} is a checkpoint dict')
    torch.save({'model_state_dict': state_dict}, model_path)

    print(f'Evaluating on saved model')
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']

    model = ResStackBiLSTMMaxout(emb=emb, padding_idx=vocab[PAD])
    model.load_state_dict(state_dict=model_state_dict)
    model.to(device)

    acc = predict(model, test_loader, device)
    print(f'Test acc: {acc:.8f}')