
import argparse
import random
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# from data import PAD, TagDataset, load_pretrained_embeds
from models import BiLSTM

def train():
    # criterion = nn.CrossEntropyLoss(ignore_index=y_pad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.999))
    pass

if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('repr', metavar='repr', type=str, help='one of a,b,c,d')
    # parser.add_argument('trainFile', type=str, help='input file to train on')
    # parser.add_argument('modelFile', type=str, help='file to save the model')
    # parser.add_argument("--devFile", dest='dev_path', type=str, help='dev file to calc acc during train')
    # args = parser.parse_args()
    # method = args.repr
    # train_path = args.trainFile
    # dev_path = args.dev_path
    # model_path = args.modelFile
    # vec_path = args.vec_path
    # vocab_path = args.vocab_path
    # batch_size = args.batch_size
    # print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running device: {device}')

    # tokens2ids, pretrained_vecs = None, None
    # if method == 'a':
    #     token_level, pre_suf_level, char_level = True, False, False
    # elif method == 'b':
    #     token_level, pre_suf_level, char_level = False, False, True
    # elif method == 'c':
    #     token_level, pre_suf_level, char_level = True, True, False
    #     if vec_path is not None and vocab_path is not None:
    #         tokens2ids, pretrained_vecs = load_pretrained_embeds(vec_path, vocab_path)
    # elif method == 'd':
    #     token_level, pre_suf_level, char_level = True, False, True
    #
    # train_dataset = TagDataset(train_path, return_y=True, tokens2ids=tokens2ids)
    # token_pad = train_dataset.tokens2ids[PAD]
    # pre_pad = train_dataset.pre2ids[PAD]
    # suf_pad = train_dataset.suf2ids[PAD]
    # char_pad = train_dataset.char2ids[PAD]
    # y_pad = len(train_dataset.tags2ids)
    # o_id = train_dataset.tags2ids['O'] if 'O' in train_dataset.tags2ids else y_pad
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                           collate_fn=lambda b: pad_collate(b, token_pad, pre_pad, suf_pad, char_pad, y_pad))
    #
    # if dev_path:
    #     dev_dataset = TagDataset(dev_path, return_y=True,
    #                              tokens2ids=train_dataset.tokens2ids,
    #                              char2ids=train_dataset.char2ids,
    #                              pre2ids=train_dataset.pre2ids,
    #                              suf2ids=train_dataset.suf2ids,
    #                              tags2ids=train_dataset.tags2ids)
    #     dev_loader = DataLoader(dev_dataset, batch_size=500, shuffle=False,
    #                             collate_fn=lambda b: pad_collate(b, token_pad, pre_pad, suf_pad, char_pad, y_pad))
    # else:
    #     dev_loader = None
    #
    # model = BiLSTM(vocab_size=train_dataset.vocab_size,
    #                      pre_vocab_size=train_dataset.pre_vocab_size,
    #                      suf_vocab_size=train_dataset.suf_vocab_size,
    #                      alphabet_size=train_dataset.alphabet_size,
    #                      tagset_size=train_dataset.tagset_size,
    #                      token_padding_idx=token_pad,
    #                      pre_padding_idx=pre_pad,
    #                      suf_padding_idx=suf_pad,
    #                      char_padding_idx=char_pad,
    #                      token_level=token_level,
    #                      pre_suf_level=pre_suf_level,
    #                      char_level=char_level,
    #                      pretrained_vecs=pretrained_vecs)
    # model.to(device)
    # # best_acc, accuracies, steps, state_dict = train(model, train_loader, dev_loader, device, y_pad, o_id, model_path)
    #
    # print(f'steps = {steps}')
    # print(f'method_{method} = {accuracies}')
    # print(f'method_{method} best_acc_dev: {best_acc}')
    #
    # print(f'Saving model and train dataset to: {model_path} is a checkpoint dict')
    # torch.save({
    #     'model_state_dict': state_dict,
    #     'tokens2ids': train_dataset.tokens2ids,
    #     'char2ids': train_dataset.char2ids,
    #     'pre2ids': train_dataset.pre2ids,
    #     'suf2ids': train_dataset.suf2ids,
    #     'tag2ids': train_dataset.tags2ids,
    # }, model_path)