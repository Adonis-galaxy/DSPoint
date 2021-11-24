from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import ModelNet40
from model import DSPoint
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import random
import time 

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda")

    model = DSPoint(args).to(device)
    print(str(model))
    model = nn.DataParallel(model)


    # opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                min_lr=0.00001)


    criterion = cal_loss
    best_test_acc = 0
    best_test_epoch=0
    best_avg_test_acc=0
    debug_times=10

    print("Start training! Good Luck!")
    for epoch in range(args.epochs):
        epoch_start_time=time.time()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0
        iter=0
        for data, label in (train_loader):
            # show the training speed

            if debug_times>=0:
                if iter==10:
                    print("Congratulations!")
                    print("Training starts successfully!")
                debug_times-=1

            data, label = data.to(device), label.to(device).squeeze() 
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()

            start_time = time.time()
            logits = model(data)
            loss = criterion(logits, label)
            
            loss.backward() 
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1
            iter+=1

        # scheduler
        scheduler.step(train_loss)

        print ('Train time is %.2fs' % total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.2f, train acc: %.2f%%, train avg acc: %.2f%%' % (epoch,
                                                                                train_loss*1.0/count,
                                                                                metrics.accuracy_score(
                                                                                train_true, train_pred)*100,
                                                                                metrics.balanced_accuracy_score(
                                                                                train_true, train_pred)*100)
        io.cprint(outstr)
        
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            start_time = time.time()
            logits = model(data)
            end_time = time.time()
            total_time += (end_time - start_time)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        print ('test total time is %.2fs' % total_time)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.2f, test acc: %.2f%%, test avg acc: %.2f%%' % (epoch,
                                                                            test_loss*1.0/count,
                                                                            test_acc*100,
                                                                            avg_per_class_acc*100)
        io.cprint(outstr)
        epoch_end_time=time.time()
        epoch_time=epoch_end_time-epoch_start_time
        io.cprint('Epoch Time: %d s' % epoch_time)
        # save model of highest eval acc
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_test_epoch=epoch
            best_avg_test_acc=avg_per_class_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
        
    outstr = 'Best test acc: %.2f%%, test avg acc: %.2f%%, epoch: %d' % (best_test_acc*100,best_avg_test_acc*100,best_test_epoch)
                                                                        
    io.cprint(outstr)

def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda")

    model = DSPoint(args).to(device)
    model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1] 
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.4f, test avg acc: %.4f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    # Testing settings
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path, for testing')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    # Training settings
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--L', type=int, default=10,
                        help='L of high frequency encode function')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    ## main init ##
    args = parser.parse_args()
    _init_()
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    io.cprint(
        'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    if not args.eval:
        train(args, io)
    else:
        test(args, io)
