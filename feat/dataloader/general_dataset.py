import os.path as osp
import os
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import glob




class General_dataset(Dataset):

    def __init__(self, args):
        
        

        def get_imgpath_from_dir(dir_path):
            return glob.glob(glob.escape(dir_path)+'/*.png') + glob.glob(glob.escape(dir_path)+'/*.jpg') + glob.glob(glob.escape(dir_path)+'/*.JPG')

        self.data_dir_path = args.data_path
        root_path, dirnames, _ = next(os.walk(self.data_dir_path))
        dirnames.remove('Query')
        label_names = [ dirname for dirname in dirnames]
        query = get_imgpath_from_dir(osp.join(root_path,'Query'))
        support = []; class_lens = []; support_labels = []

        for i,dirname in enumerate(dirnames):
            img_paths = get_imgpath_from_dir(osp.join(root_path,dirname))
            support += img_paths
            class_lens.append(len(img_paths))
            support_labels += len(img_paths)*[i]  # i: label


        self.data = query + support
        self.query = query
        self.support = support
        self.num_class = len(label_names)
        self.label_names = label_names # str label
        self.support_labels = support_labels # int label
        self.class_lens = class_lens
        self.vis = args.vis


        if args.model_type == 'AmdimNet':
            INTERP = 3
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            self.transform = transforms.Compose([
                transforms.Resize(146, interpolation=INTERP),
                transforms.CenterCrop(128),
                post_transform
            ])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            self.transform = transforms.Compose([
                transforms.Resize(84, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                normalize])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        label = self.label_names[self.support_labels[i-len(self.query)]] if i > len(self.query) else '' # can't be None!!!
        transformed_img = self.transform(Image.open(path).convert('RGB'))
        return transformed_img, path, label            

