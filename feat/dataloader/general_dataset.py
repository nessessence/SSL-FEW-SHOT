import os.path as osp
import os
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import glob


# THIS_PATH = osp.dirname(__file__)
# ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
# IMAGE_PATH = osp.join(ROOT_PATH, 'data/cub/images')
# SPLIT_PATH = osp.join(ROOT_PATH, 'data/cub/split')

# This is for the CUB dataset, which does not support the ResNet encoder now
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)
class General_dataset(Dataset):

    def __init__(self, args):
        
        

        def get_imgpath_from_dir(dir_path):
            return glob.glob(glob.escape(dir_path)+'/*.png') + glob.glob(glob.escape(dir_path)+'/*.jpg')

        # self.data_dir_path = osp.join( osp.dirname(__file__), args.data_path )
        self.data_dir_path = args.data_path
        root_path, dirnames, _ = next(os.walk(self.data_dir_path))
        dirnames.remove('Query')
        label_names = [ dirname for dirname in dirnames]
        # labels = [i for i in range(len(label_names))]
        query = get_imgpath_from_dir(osp.join(root_path,'Query'))
        gallery = []; class_lens = []
        for dirname in dirnames:
            class_label = get_imgpath_from_dir(dirname)
            gallery += class_label
            class_lens.append(len(class_label))

        # all_img_paths = glob.glob(glob.escape(self.data_dir_path)+'/**/*.png' , recursive=True)+glob.glob(glob.escape(self.data_dir_path)+'/**/*.jpg' , recursive=True)
        # query_img_paths =  glob.glob(glob.escape(self.data_dir_path)+'/query/*.png' , recursive=True)+glob.glob(glob.escape(self.data_dir_path)+'/query/*.jpg' , recursive=True)
        # gallery_img_paths =[ img_path for img_path in query_img_paths if img_path not in query_img_paths ]
        # if label: gallery_img_label = [ osp.basename(osp.dirname(img_path)) for img_path in gallery_img_paths]

        self.data = query + gallery
        self.query = query
        self.gallery = gallery
        self.num_class = len(label_names)
        self.label_names = label_names
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
        path, label = self.data[i], self.label[i]
        transformed_img = self.transform(Image.open(path).convert('RGB'))
        if self.vis: return transformed_img, path, label   
        return transformed_img, label            

