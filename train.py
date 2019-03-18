import torch
import torch.nn as nn
from seq2seq import seq2seq
import numpy as np
import time
import shutil
import pickle
import SBT_encode
import re

MAX_LEN = 2200


class Evaluation():
    def __init__(self):
        self.epoch = 1
        self.loss = .0
        self.count_data = 0
        self.count_save = 0
        self.count_chunk = 0
        self.history = {}

    def reset(self, epoch):
        self.epoch = epoch
        self.loss = .0
        self.count_data = 0
        self.count_save = 0
        self.count_chunk = 0
        self.history[epoch] = []

    def __call__(self, loss, outputs):
        loss_ = loss.cpu().detach().numpy()
        outputs_ = outputs.cpu().detach().numpy().squeeze()
        chunk_size = outputs_.shape[0]
        self.loss += loss_ * chunk_size
        self.count_data += chunk_size
        self.count_chunk += 1

    def avg_loss(self):
        return self.loss / self.count_data

    def save(self, train_loss, val_loss):
        self.count_save += 1
        self.history[self.epoch].append((train_loss, val_loss))


def preprocessing(file_name, max_len_code=6800, max_len_comment=13000):
    # load data
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    comment = []
    code = []
    for i in range(len(data)):
        temp_comment, temp_code = data[i]
        comment.append(temp_comment)
        code.append(temp_code)

    commment_wordlist = []
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    for i in range(len(comment)):
        temp_list = re.split(pattern, comment[i])
        for x in temp_list:
            commment_wordlist.append(x)
    temp_dict = {}
    for word in commment_wordlist:
        temp_dict[word] = commment_wordlist.count(word)
    temp_wordlist = sorted(temp_dict.items(), key=lambda kv: (-kv[1], kv[0]))[:3000]
    commment_wordlist = [temp_wordlist[i][0] for i in range(len(temp_wordlist))]

    comment_dict = dict(zip(commment_wordlist, range(len(commment_wordlist))))

    encoder = SBT_encode.Encoder()

    code_in_num = []
    comment_in_num = []

    for i in range(len(code)):
        code_in_num.append(encoder.encode(code[i]))

    for i in range(len(comment)):
        split_list = re.split(pattern, comment[i])
        temp_list = []
        for x in split_list:
            if x in comment_dict:
                temp_list.append(comment_dict[x])
            else:
                temp_list.append(0)
        comment_in_num.append(temp_list)

    # len_code = max([len(code_in_num[i]) for i in range(len(code_in_num))])
    # len_comment = max([len(comment_in_num[i]) for i in range(len(comment_in_num))])
    # max_len_code = max([max(sublist) for sublist in code_in_num])
    # max_len_comment = max([max(sublist) for sublist in comment_in_num])
    print("Max length of code: " + str(max_len_code))
    print("Max length of comment: " + str(max_len_comment))

    for i in range(len(code_in_num)):
        while len(code_in_num[i]) < MAX_LEN:
            code_in_num[i].append(0)

    for i in range(len(comment_in_num)):
        while len(comment_in_num[i]) < MAX_LEN:
            comment_in_num[i].append(0)

    return code_in_num, comment_in_num, max_len_code, max_len_comment


def build_model(word_size_encoder, word_size_decoder, emb_dim=10, hidden_size=100, learning_rate=0.1, device=None):
    model = seq2seq(word_size_encoder, emb_dim, hidden_size, word_size_decoder, MAX_LEN)
    # run on the gpu or cpu
    model = model.to(device)

    criterion = nn.L1Loss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer


def train_model(model, criterion, optimizer, dataloaders,
                num_epochs=1, best_loss=10,
                evaluate=Evaluation(), device=None):
    # init timer
    since = time.time()
    start_epoch = evaluate.epoch
    step = 500
    # if istest: step = 10

    for epoch in range(start_epoch, num_epochs + 1):
        print('\nEpoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        ## reset evaluator in a new epoch
        evaluate.reset(epoch)

        for i in range(len(dataloaders['train'][0])):

            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            # inputs, targets = inputs.to(device), targets.to(device)

            inputs, targets = dataloaders['train'][0][i], dataloaders['train'][1][i]
            inputs = torch.LongTensor(inputs)
            targets = torch.LongTensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)

            model.zero_grad()

            # regular stuff
            outputs, _ = model(inputs, targets)
            # squeeze the unnecessary batchsize dim
            loss = criterion(outputs[0], targets.float())
            loss.backward()
            optimizer.step()
            print(loss)

            # evaluation
            evaluate(loss, outputs)

            # validate every n chunks
            if i % step == 0:
                train_loss = evaluate.avg_loss()
                # validate first
                val_loss = validate_model(model, criterion,
                                          dataloaders['val'],
                                          device=device)

                # update best loss
                is_best = val_loss < best_loss
                best_loss = min(val_loss, best_loss)

                # verbose
                print('[%i] '
                      'train-loss: %.4f '
                      'val-loss: %.4f '
                      '' % (evaluate.count_save,
                            train_loss,
                            val_loss))

                # save for plot
                evaluate.save(train_loss, val_loss)
                save_checkpoint({'model': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'best_loss': best_loss,
                                 'history': evaluate}, is_best)

            # if istest:
            #     if i == 100: break

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# could also be use to test
def validate_model(model, criterion, loader, device=None, verbose=False):
    model.eval()  # Set model to evaluate mode

    evaluate = Evaluation()
    step = 50
    # if istest: step = 1

    with torch.no_grad():
        for i in range(len(loader[0])):
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            inputs, targets = loader[0][i], loader[1][i]
            inputs = torch.LongTensor(inputs)
            targets = torch.LongTensor(targets)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, _ = model(inputs, targets)
            loss = criterion(outputs[0], targets.float())
            evaluate(loss, outputs)

            if verbose:
                if i % step == 0:
                    print('[%i] val-loss: %.4f' % (i, evaluate.avg_loss()))

            # if istest:
            #     if j == 2: break

    model.train()  # Set model to training mode
    return evaluate.avg_loss()


def save_checkpoint(state, is_best):
    filename = 'checkpoint' + str(model_num) + '.pth.tar'
    bestname = 'model_best' + str(model_num) + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def check_cuda():
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()
    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        device = torch.device("cuda")
        extras = {"num_workers": 1, "pin_memory": True}
    else:  # Otherwise, train on the CPU
        device = torch.device("cpu")
        extras = False
    return use_cuda, device, extras


def main(learning_rate=0.01, hidden_size=100, device=None):
    # hyperparameters
    num_epochs = 50
    learning_rate = 0.05
    # hidden_size = 100

    print('------- Hypers --------\n'
          '- epochs: %i\n'
          '- learning rate: %g\n'
          '- hidden size: %i\n'
          '----------------'
          '' % (num_epochs, learning_rate, hidden_size))

    train_code_in_num, train_comment_in_num, train_word_size_encoder, train_word_size_decoder = preprocessing(
        'data/train.pkl')
    val_code_in_num, val_comment_in_num, val_word_size_encoder, val_word_size_decoder = preprocessing('data/valid.pkl')
    test_code_in_num, test_comment_in_num, test_word_size_encoder, test_word_size_decoder = preprocessing(
        'data/test.pkl')

    dataloaders = {}
    dataloaders['train'] = (train_code_in_num, train_comment_in_num)
    dataloaders['val'] = (val_code_in_num, val_comment_in_num)
    dataloaders['test'] = (test_code_in_num, test_comment_in_num)

    # save loader and encoder for later use
    # torch.save({'loaders': dataloaders,
    #             'encoder': encoder,
    #             'hidden_size': hidden_size},
    #            'init' + str(model_num) + '.pth.tar')

    model, criterion, optimizer = build_model(train_word_size_encoder, train_word_size_decoder,
                                              emb_dim=10, hidden_size=100, learning_rate=0.1, device=device)
    evaluate = Evaluation()

    train_model(model, criterion, optimizer, dataloaders,
                num_epochs=num_epochs, evaluate=evaluate,
                best_loss=10, device=device)


if __name__ == "__main__":
    model_num = 0
    use_cuda, device, extras = check_cuda()
    main(device=device)
