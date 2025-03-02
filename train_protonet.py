import argparse
import os.path as osp
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from feat.dataloader.samplers import CategoriesSampler
from feat.models.protonet import ProtoNet
from feat.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval
from tensorboardX import SummaryWriter

from feat.logger.logging import Logger
import sys



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--model_type', type=str, default='ConvNet', choices=['ConvNet', 'ResNet', 'AmdimNet'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])
    # MiniImageNet, ConvNet, './saves/initialization/miniimagenet/con-pre.pth'
    # MiniImageNet, ResNet, './saves/initialization/miniimagenet/res-pre.pth'
    # CUB, ConvNet, './saves/initialization/cub/con-pre.pth'
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--gpu', default='0')

    # AMDIM Modelrd
    parser.add_argument('--ndf', type=int, default=256)
    parser.add_argument('--rkhs', type=int, default=2048)
    parser.add_argument('--nd', type=int, default=10)

    parser.add_argument('--trlog_checkpoint', type=str, default=None)
    parser.add_argument('--model_checkpoint', type=str, default=None)

    parser.add_argument('--load_train_checkpoint',action='store_true')


    args = parser.parse_args()
    sys.stdout = Logger(osp.join(args.save_path, 'log.txt'),append=args.load_train_checkpoint)
    pprint(vars(args))

    set_gpu(args.gpu)


    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from feat.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from feat.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from feat.dataloader.tiered_imagenet import tieredImageNet as Dataset       
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, 100, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 500, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    
    model = ProtoNet(args)
    if args.model_type == 'ConvNet':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.model_type == 'ResNet':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    elif args.model_type == 'AmdimNet':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        
    
    # load pre-trained model (no FC weights)

    args.out_name = osp.basename(osp.dirname(args.save_path))



    model_dict = model.state_dict()

    if args.init_weights is not None:
        model_detail = torch.load(args.init_weights)
        if  isinstance(model_detail, dict):
            if 'params' in model_detail:
                pretrained_dict = model_detail['params']
                # remove weights for FC
                pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                print(pretrained_dict.keys())
                model_dict.update(pretrained_dict)
            else:
                pretrained_dict = model_detail['model']
                pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
                model_dict.update(pretrained_dict)
            
        else: 
            model_dict = model_detail
    model.load_state_dict(model_dict)    
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    def save_model(flag):
        if not os.path.isdir( os.path.join(args.save_path) ):
            print("make new directory:")
            os.mkdir(os.path.join(args.logs_dir, args.save_path))
        torch.save(model.encoder.state_dict(), osp.join(args.save_path, f"modelEncoder_{args.out_name}_{flag}.pth")) # Note: save "encoder" eg AmdimNet, not the protoNet
        if flag == 'epoch_last':
            torch.save(trlog, osp.join(args.save_path,f"trainlog_{args.out_name}_{flag}.pth"))
            torch.save(optimizer.state_dict(),osp.join(args.save_path,f"optimizer_{args.out_name}_{flag}.pth"))
        print(f"save checkpoint [{flag}] at {args.save_path}")

    

    if args.load_train_checkpoint:
        flag = 'epoch_last'
        if osp.isdir(osp.join(args.save_path)) and not (args.trlog_checkpoint and args.model_checkpoint ):
            encoder_weight = torch.load(osp.join(args.save_path, f"modelEncoder_{args.out_name}_{flag}.pth"))
            trlog = torch.load(osp.join(args.save_path,f"trainlog_{args.out_name}_{flag}.pth"))
            optimizer.load_state_dict(torch.load(osp.join(args.save_path,f"optimizer_{args.out_name}_{flag}.pth")))
        else:
            trlog = torch.load(args.trlog_checkpoint)
            encoder_weight = torch.load(args.model_checkpoint)
        model.encoder.load_state_dict(encoder_weight)
        print("load training checkpoint complete")
        
    
    else:
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        trlog['cur_train_epoch'] = 0
        print("start new training")

    start_epoch = trlog['cur_train_epoch']+1
    print(f"start training from epoch: {start_epoch}")

    timer = Timer()
    global_count = 0
    writer = SummaryWriter(logdir=args.save_path)
    
    for epoch in range(start_epoch, args.max_epoch + 1):
        lr_scheduler.step()
        model.train()
        tl = Averager()
        ta = Averager()

        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            
        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.way
            data_shot, data_query = data[:p], data[p:]
            logits = model(data_shot, data_query)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        label = torch.arange(args.way).repeat(args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
            
        print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        print('evaluating:')
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = args.shot * args.way
                data_shot, data_query = data[:p], data[p:]
    
                logits = model(data_shot, data_query)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)    
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/val_loss', float(vl), epoch)
        writer.add_scalar('data/val_acc', float(va), epoch)        
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        trlog['cur_train_epoch'] = epoch
        save_model('epoch_last')

        

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()

    # Test Phase
    flag = 'epoch_last'
    trlog = torch.load(osp.join(args.save_path, f"trainlog_{args.out_name}_{flag}.pth"))
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, 10000, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((10000,))

    model.encoder.load_state_dict(torch.load(osp.join(args.save_path, f"modelEncoder_{args.out_name}_{flag}.pth")))
    model.eval()

    ave_acc = Averager()
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
        
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]
    
            logits = model(data_shot, data_query)
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    print('Val Best Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc'], ave_acc.item()))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
