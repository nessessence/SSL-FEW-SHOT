import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from feat.dataloader.samplers import CategoriesSampler
from feat.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval
from feat.dataloader.cub import CUB as Dataset
import matplotlib.pyplot as plt


from feat.networks.amdimnet import AmdimNet
from feat.dataloader.general_dataset import General_dataset as Dataset # general dataset!
from torch.utils.data import DataLoader
from tqdm import tqdm

from feat.utils import euclidean_metric


def load_checkpoint(args):
    checkpoint_dir = osp.join(args.logs_dir, args.out_name)
    assert os.path.isdir(checkpoint_dir)
    features_dict = torch.load(osp.join(checkpoint_dir,'feature_dict.pt'))
    labels_dict = torch.load(osp.join(checkpoint_dir,'labels_dict.pt'))
    dataset = torch.load(osp.join(checkpoint_dir,'dataset.pt'))
    
    return features_dict,labels_dict,dataset

def save_checkpoint(args,features_dict,labels_dict,dataset):
    checkpoint_dir = osp.join(args.logs_dir, args.out_name)
    if not os.path.isdir( checkpoint_dir ):
              print(f"make new log directory: {checkpoint_dir} ")
              os.mkdir(checkpoint_dir)
              
    torch.save(features_dict,osp.join(checkpoint_dir,'feature_dict.pt'))
    torch.save(labels_dict,osp.join(checkpoint_dir,'labels_dict.pt'))
    torch.save(dataset,osp.join(checkpoint_dir,'dataset.pt'))

    


def main():
    args = parser.parse_args() 
    if not args.out_name:
        args.out_name = osp.basename(osp.dirname(args.data_path))
    pprint(vars(args))
    if  args.load_checkpoint and os.path.isdir(osp.join(args.logs_dir, args.out_name)): 
        features_dict,labels_dict,dataset = load_checkpoint(args)
    else:
        print("start new process")
        print(args.load_checkpoint,os.path.isdir(osp.join(args.logs_dir, args.out_name)))
        print("model init...")
        model_weight = torch.load('./pretrained/imagenet1K_ndf192_rkhs_1536_rd8_ssl_cpt.pth')
        model = AmdimNet(ndf=args.ndf, n_rkhs=args.rkhs, n_depth=args.nd)
        model_dict = model.state_dict()
        pretrained_dict = model_weight['model']
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
        model = model.cuda()

        print("dataset init...")
        dataset = Dataset(args)
        dataloader = DataLoader(dataset,batch_size=args.batch_size,pin_memory=True) 
        features_dict = {}; labels_dict = {} 

        print("extracting features")
        with torch.no_grad():
            for i,batch in tqdm(enumerate(dataloader,1)):
                transformed_imgs, paths, labels = batch
                transformed_imgs = transformed_imgs.cuda()
                features_batch = model(transformed_imgs)
                for path,feature,label in zip(paths,features_batch,labels):
                    features_dict[path] = feature.cpu()
                    labels_dict[path] = label if label else None # if from query: label == ''

        save_checkpoint(args,features_dict,labels_dict,dataset)
        print("saving checkpoints log complete")

    support_features = torch.stack([features_dict[path] for path in dataset.support ])
    query_features = torch.stack([features_dict[path] for path in dataset.query])
    print(torch.tensor(dataset.class_lens))
    indices = torch.cumsum(torch.tensor(dataset.class_lens),dim=0)
    indices = torch.cat([torch.tensor([0]),indices]) 
    mean_support_featers =  torch.stack([torch.mean(support_features[ indices[i]:indices[i+1] ],dim=0) for i in range(len(indices)-1)])
    logits = euclidean_metric(query_features, mean_support_featers) / args.temperature # distance here is negative --> similarity 
    print(logits.shape)
    return logits,dataset




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
    parser.add_argument('--vis',action='store_true')


    # modify
    parser.add_argument('--data_path', type=str, default="./data/demo_pet_dataset/")
    parser.add_argument('--batch_size', type=int, default= 512)

    parser.add_argument('--load_checkpoint',action='store_true')
    

    # working_dir = osp.dirname(osp.abspath(__file__))
    # parser.add_argument('--logs_dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='./logs')
    parser.add_argument('--out_name', type=str, default=None)
    
    main()
