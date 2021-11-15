import os
import time
import datetime

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from termcolor import colored

from dataset.parallel_sampler import ParallelSampler
from train.utils import named_grad_param, grad_param, get_norm
from train.fgm import FGM
from train.ici import ICI
import joblib
from train.vis import createHTML
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def train(train_data, val_data, model, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
                                  os.path.curdir,
                                  "tmp-runs",
                                  str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    best_std = 0
    sub_cycle = 0
    best_path = None

    opt = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=args.lr,  weight_decay=0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'max', patience=args.patience//2, factor=0.1, verbose=True)

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    train_gen = ParallelSampler(train_data, args, args.train_episodes)
    train_gen_val = ParallelSampler(train_data, args, args.val_episodes)
    val_gen = ParallelSampler(val_data, args, args.val_episodes)

    train_accs,val_accs = [0],[0]

    ebd_state = False
    rnn_state = True
    frozen_change_time = 0

    for ep in range(args.train_epochs):

        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'ebd': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                    ncols=80, leave=False, desc=colored('Training on train',
                        'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            train_one(task, model, opt, args, grad)

        

        if ep % 10 == 0:
            acc, std = test(train_data, model, args, args.val_episodes, False,
                            train_gen_val.get_epoch())

            train_accs.append(acc)
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)

        # Evaluate validation accuracy
        cur_acc, cur_std = test(val_data, model, args, args.val_episodes, True,
                                val_gen.get_epoch())
        val_accs.append(cur_acc)
        # print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
        #        "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}").format(
        #        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
        #        "ep", ep,
        #        colored("val  ", "cyan"),
        #        colored("acc:", "blue"), cur_acc, cur_std,
        #        colored("train stats", "cyan"),
        #        colored("ebd_grad:", "blue"), np.mean(np.array(grad['ebd'])),
        #        colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
        #        ), flush=True)

        # Update the current best model if val acc is better
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_std = cur_std
            best_path = os.path.join(out_dir, str(ep))

            # save current model
            print("{}, Save cur best model to {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                best_path))

            torch.save(model['ebd'].state_dict(), best_path + '.ebd')
            torch.save(model['clf'].state_dict(), best_path + '.clf')

            sub_cycle = 0
        else:
            sub_cycle += 1

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break
            # if frozen_change_time == 1:
            #     break
            # print('****************************************')
            # print('exchange grad frozen...')
            # print('****************************************')
            # model['ebd'].change_ebd_frozen(True)
            # model['ebd'].change_rnn_frozen(True)
            # # ebd_state = not ebd_state
            # # rnn_state = not rnn_state
            # sub_cycle = 0
            # frozen_change_time += 1
            # if frozen_change_time == 5:
            #     break

    print("{}, End of training. Restore the best weights".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')),
            flush=True)

    print("Best acc: {:.4f} ± {:.4f}".format(best_acc, best_std),flush=True)

    joblib.dump([train_accs,val_accs],'lstm_accs.pkl')

    # x=list(range(len(train_accs)))
    # plt.plot(x,train_accs,label='Train Accuracy')
    # plt.plot(x,val_accs,label='Validation Accuracy')
    # plt.ylim([0,1])
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.savefig('meta.png')

    # restore the best saved model
    model['ebd'].load_state_dict(torch.load(best_path + '.ebd'))
    model['clf'].load_state_dict(torch.load(best_path + '.clf'))

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      str(int(time.time() * 1e7))))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print("{}, Save best model to {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            best_path), flush=True)

        torch.save(model['ebd'].state_dict(), best_path + '.ebd')
        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def train_one(task, model, opt, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['ebd'].train()
    model['clf'].train()
    opt.zero_grad()
    support, query = task
    # Embedding the document
    XS = model['ebd'](support)
    YS = support['label']
    XQ = model['ebd'](query)
    YQ = query['label']
    # Apply the classifier
    _, loss = model['clf'](XS, YS, XQ, YQ)

    if loss is not None:
        loss.backward()
        # fgm.attack()
        # XS_attack = model['ebd'](support)
        # XQ_attack = model['ebd'](query)
        # _, loss_adv = model['clf'](XS_attack, YS, XQ_attack, YQ)
        # loss_adv.backward()  # ??????????grad?????????????
        # fgm.restore()  # ??embedding??

    if torch.isnan(loss):
        # do not update the parameters if the gradient is nan
        # print("NAN detected")
        # print(model['clf'].lam, model['clf'].alpha, model['clf'].beta)
        return

    if args.clip_grad is not None:
        nn.utils.clip_grad_value_(grad_param(model, ['ebd', 'clf']),
                                  args.clip_grad)

    grad['clf'].append(get_norm(model['clf']))
    grad['ebd'].append(get_norm(model['ebd']))

    opt.step()


def test(test_data, model, args, num_episodes,  return_score=False, vocab=None, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['ebd'].eval()
    model['clf'].eval()


    if sampled_tasks is None:
        sampled_tasks = ParallelSampler(test_data, args,
                                        num_episodes).get_epoch()

    acc1s = []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))

    for task in sampled_tasks:
        acc1 = test_one(task, model, args, return_score=return_score, vocab=vocab)
        acc1s.append(acc1)

    acc1s = np.array(acc1s)

    if verbose:
        print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(acc1s),
                colored("std", "blue"),
                np.std(acc1s),
                ), flush=True)
    return np.mean(acc1s), np.std(acc1s)

html_index = 0



index2label = joblib.load('index2label.pkl')

def test_one(task, model, args, return_score=False, vocab=None):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task
    YS = support['label']
    global html_index

    XS = model['ebd'](support)
   
    

    XQ = model['ebd'](query)
    YQ = query['label']

    YS, YQ = model['clf'].reidx_y(YS, YQ)

    # Apply the classifier
    acc1, _ = model['clf'](XS, YS, XQ, YQ)

    return acc1
