# encoding:utf-8
import os, sys, numpy as np, random, time, cv2
import torch

import jpeg4py as jpeg
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
from glob import glob
ia.seed(random.randint(1, 10000))


class ClassifyDataset(data.Dataset):
    def __init__(self, base_data_path, train, transform, id_name_path,  device, little_train=False, with_file_path=False, origin_img_size=(1920, 1080), input_size=224, C = 2048, test_mode=False):
        print('data init')
        
        self.train = train
        self.base_data_path=base_data_path
        self.transform=transform
        self.fnames = []
        self.resize = input_size
        self.little_train = little_train
        self.id_name_path = id_name_path
        self.C = C
        self.origin_img_size = origin_img_size
        self.device = device
        self._test = test_mode
        self.with_file_path = with_file_path
        self.img_augsometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.augmentation = iaa.Sequential(
            [
                # augment without change bboxes 
                self.img_augsometimes(
                    iaa.SomeOf((1, 2), [
                        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
                        iaa.Sharpen((0.1, .8)),       # sharpen the image
                        # iaa.GaussianBlur(sigma=(2., 3.5)),
                        iaa.OneOf([
                            iaa.GaussianBlur(sigma=(2., 3.5)),
                            iaa.AverageBlur(k=(2, 5)),
                            iaa.BilateralBlur(d=(7, 12), sigma_color=(10, 250), sigma_space=(10, 250)),
                            iaa.MedianBlur(k=(3, 7)),
                        ]),
                        

                        iaa.AddElementwise((-50, 50)),
                        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                        iaa.JpegCompression(compression=(80, 95)),

                        iaa.Multiply((0.5, 1.5)),
                        iaa.MultiplyElementwise((0.5, 1.5)),
                        iaa.ReplaceElementwise(0.05, [0, 255]),
                        # iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                        #                 children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                        iaa.OneOf([
                            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                            children=iaa.WithChannels(1, iaa.Add((-10, 50)))),
                            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                            children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                        ]),

                        # iaa.Affine(
                        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        #     rotate=(-25, 25),
                        #     shear=(-8, 8)
                        # )

                    ], random_order=True)
                ),

                iaa.Fliplr(.5),
                iaa.Flipud(.25),

            ],
            random_order=True
        )

        self.fnames = self.get_data_list(base_data_path)
        self.num_samples = len(self.fnames)

        self.get_id_map()
    
    def get_id_list(self):
        id_set = set()
        if isinstance(self.base_data_path, list):
            for i in self.base_data_path:
                id_tl = os.listdir(i)
                for j in id_tl:
                    id_set.add(j)
        else:
            id_tl = os.listdir(self.base_data_path)
            for j in id_tl:
                id_set.add(j)
        return list(id_set)


    def get_id_map(self):
        self.id_name_map = {}
        self.name_id_map = {}
        if not os.path.exists(self.id_name_path):
            id_list = self.get_id_list()
            with open(self.id_name_path, 'w') as f:
                for it, cls_name in enumerate(id_list):
                    self.name_id_map[cls_name] = it
                    self.id_name_map[it] = cls_name
                    f.write(cls_name+'\n')
        else:
            with open(self.id_name_path, 'r') as f:
                itt = 0
                for line in f:
                    self.name_id_map[line.strip()] = itt
                    self.id_name_map[itt] = line.strip()
                    itt += 1

    def get_data_list(self, base_data_path):
        cls_file_list = []
        if isinstance(base_data_path, list):
            for i in base_data_path:
                cls_file_list = cls_file_list + glob(i + '/*/*.jpg')
        else:
            cls_file_list = glob(base_data_path + '/*/*.jpg')
        if self.little_train:
            return cls_file_list[:self.little_train]
        return cls_file_list
    
    def get_label_from_path(self, in_path):
        t_str = in_path.split('/')[-2]
        return self.name_id_map[t_str]

    def __getitem__(self,idx):
        
        fname = self.fnames[idx]
        if self._test:
            print(fname)
        # img = cv2.imread(fname)
        img = jpeg.JPEG(fname).decode()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        h, w, c = img.shape 
        h = float(h/self.origin_img_size[1])
        w = float(w/self.origin_img_size[0])
        assert img is not None, print(fname)
        label = self.get_label_from_path(fname)
        
        if self.train:
            # add data augument
            seq_det = self.augmentation.to_deterministic()
            img = seq_det.augment_images([img])[0]
 
        img = self.transform(img)
        
        if self.with_file_path:
            return img, label, fname

        return img, label

    def __len__(self):
        return self.num_samples

    
if __name__ == "__main__":

    from data_utils import *

    transform = transforms.Compose([
        transforms.Lambda(padding_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = ClassifyDataset(base_data_path='/data/datasets/truth_data/classify_data/top100_checkout/20190612_train', train=True, transform = transform, id_name_path='/home/ubuntu/project/classify.pytorch/saved_models/densenet121_top100_224_truth_and_sync/id.txt', test_mode=True, C=900, device='cuda:0')
    train_loader = DataLoader(train_dataset, batch_size=2,shuffle=True, num_workers=0)
    train_iter = iter(train_loader)
    print(len(train_dataset))
    for i in range(200):
        img, label, w, h = next(train_iter)
        print(img.shape, label.shape, w.shape, h.shape)

        print('~'*50 + '\n\n\n')

        img = tensor2img(img, normal=True)
        cv2.imshow('img', img)
        
        print('now cls_id is ', label[0].item(), ' label is : ', train_dataset.id_name_map[label[0].item()], 'w :', w, ' h : ', h)
        if cv2.waitKey(12000)&0xFF == ord('q'):
            break


