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

def get_label(
    data_path="./data/demo_pet_dataset/",
    logs_dir="./logs",
    ndf=192,
    rkhs=1536,
    nd=8,
    batch_size=512,
    model_type='AmdimNet',
    _load_checkpoint=False,
    out_name=None
):
    # args = parser.parse_args() 
    args = argparse.Namespace(
        data_path=data_path,
        logs_dir=logs_dir,
        ndf=ndf,
        rkhs=rkhs,
        nd=nd,
        batch_size=batch_size,
        load_checkpoint=_load_checkpoint,
        out_name=out_name,
        model_type=model_type,
        temperature=1,
        vis=True
    )
    args.out_name = osp.basename(osp.dirname(args.data_path)) if not args.out_name else None
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

