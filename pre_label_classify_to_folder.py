#encoding:utf-8
import os, sys, time, numpy as np, cv2, copy, time
from glob import glob 

import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision

from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.train_utils import get_config_map
from utils.model_utils import initialize_model

# cfg_path = 'configs/classify2050c_densenet121_eval.json'
cfg_path = 'configs/classify100c_densenet121_eval.json'
config_map = get_config_map(cfg_path)


model_path = '/home/ubuntu/project/classify.pytorch/saved_models/densenet121_top100/epoch_3.pth'
device = 'cuda:0'

def load_model(model_path):
    print(model_path)
    model_ft = torch.load(model_path)
    num_ftrs = model_ft.classifier.in_features
    print(num_ftrs)
    input_size = 224
    return model_ft, input_size

def padding_resize(img, resize=224):
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    h, w, c = img.shape 
    pad_img = np.zeros((resize, resize, 3), np.uint8)
    if h > w:
        nh = resize
        rs = 1.0 * h / nh 
        nw = int(w / rs )
        img = cv2.resize(img, (nw, nh))
        # print('h > w: ', img.shape)
        pw = int((resize - nw)/2)
        pad_img[:, pw : pw+nw] = img

        
    else:
        nw = resize
        rs = 1.0 * w / nw
        nh = int(h / rs )
        img = cv2.resize(img, (nw, nh))
        # print('h < w: ', img.shape)
        ph = int((resize - nh)/2)
        pad_img[ph : ph+nh, :] = img
    
    return pad_img

def image_loader(loader, image_name, device, batch_size, img_size=224, c=3):
    batch_img = torch.zeros([batch_size, c, img_size, img_size], dtype=torch.float32)
    # print(batch_img.shape)
    for bi in range(batch_size):
        image = Image.open(image_name[bi])
        # image.show()
        image = loader(image).float()
        image = torch.tensor(image, requires_grad=True)
        image = image.unsqueeze(0)
        # print(image.shape)
        batch_img[bi] = image

    return batch_img.to(device)

def load_id_name_map(file_path):
    mp = {}
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            mp[i] = int(line.strip())
            i += 1
    return mp

data_transforms = transforms.Compose([
            transforms.Lambda(lambda img: padding_resize(img)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model_ft, input_size = load_model(model_path)
    model_ft, input_size = initialize_model(config_map['model_type'], config_map['class_number'], config_map['feature_extract'], use_pretrained=False)

    model_p = nn.DataParallel(model_ft.to(device), device_ids=config_map['gpu_ids'])
    if config_map['load_from_path']:
        model_p.load_state_dict(torch.load(config_map['load_from_path']))
    model_p.eval()

    id_map_path = 'saved_models/densenet121_top100/id.txt'
    id_name_map = load_id_name_map(id_map_path)
    base_path = '/data/datasets/truth_data/classify_data/top100_checkout/train/origin/top100_multi_truth_train_20190612/'
    out_base_path = '/data/results/temp/classify_instance/20190612_train/'

    # base_path = '/data/datasets/truth_data/classify_data/top100_checkout/train/origin/top100_val_20190612/'
    # out_base_path = '/data/results/temp/classify_instance/20190612_val/'

    if not os.path.exists(out_base_path):
        os.makedirs(out_base_path)

    file_list = glob(base_path + '*.jpg')

    batch_size = 1
    right = 0
    wrong = 0
    with torch.no_grad():
        
        # now_input =  input('batch_size : ')
        # now_input = base_path + now_input
        for now_path in tqdm(file_list):
            
            input_list = [now_path]
            origin_img = cv2.imread(now_path)
            # now_cls = get_cls_by_path(now_path)

                
            # ta = time.time()
            t = image_loader(data_transforms, input_list, device, batch_size)
            # tb = time.time()
            # print('load data use time : %.2f'%(tb - ta))
            # exit()
            output = model_p(t).cpu().detach().numpy()
            # print(output.shape)
            for j in range(output.shape[0]):
                pred = np.argmax(output[j])
                now_cls = id_name_map[pred]
                now_save_path = out_base_path + '%d/'%(now_cls)
                if not os.path.exists(now_save_path):
                    os.makedirs(now_save_path)
                save_id = len(os.listdir(now_save_path))
                cv2.imwrite('%s%d_%d.jpg'%(now_save_path, now_cls, save_id), origin_img)

                


