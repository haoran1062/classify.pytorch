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
cfg_path = 'configs/classify100c_resnext50_eval.json'
cfg_path = 'configs/classify800c_se-resnext50_512_eval.json'
config_map = get_config_map(cfg_path)
input_resize = config_map['input_size']

# model_path = '/home/ubuntu/project/classify.pytorch/saved_models/densenet121_top100/epoch_3.pth'
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

def load_id_name_map(cfg_path):
    config_map = get_config_map(cfg_path)
    file_path = config_map['id_name_txt']
    mp = {}
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            mp[i] = line.strip()
            i += 1
    return mp

def load_classify_model(cfg_path, with_wh, device='cuda:0'):

    config_map = get_config_map(cfg_path)

    model_ft, input_size = initialize_model(config_map['model_type'], config_map['class_number'], config_map['feature_extract'], use_pretrained=False)
    model_p = nn.DataParallel(model_ft.to(device), device_ids=config_map['gpu_ids'])
    model_p.load_state_dict(torch.load(config_map['load_from_path']))
    model_p.eval()
    return model_p

def get_wh(w, h, origin_img_size=(1920, 1080)):
    h = float(h/origin_img_size[1])
    w = float(w/origin_img_size[0])
    return w, h

data_transforms = transforms.Compose([
            transforms.Lambda(lambda img: padding_resize(img, resize=input_resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model_ft, input_size = load_model(model_path)
    FloatTensor = torch.cuda.FloatTensor
    model_ft, input_size = initialize_model(config_map['model_type'], config_map['class_number'], config_map['feature_extract'], use_pretrained=False)

    model_p = load_classify_model(cfg_path, True)
    model_p.eval()

    id_name_map = load_id_name_map(cfg_path)

    base_path = '/data/results/temp/classify_instance/20190618/1/'
    out_base_path = '/data/results/temp/classify_instance/20190618_train/'

    base_path = '/data/results/temp/classify_instance/201907/0705_2_instances/'
    out_base_path = '/data/results/temp/classify_instance/201907/cls_out/0705_2_pre_cls/'

    base_path = '/data/results/temp/classify_instance/201907/badcase_instances_20190726/'
    out_base_path = '/data/results/temp/classify_instance/201907/cls_out/badcase_pre_cls/'


    if not os.path.exists(out_base_path):
        os.makedirs(out_base_path)
    folder_list = os.listdir(base_path)
    it = 0
    for now_folder in tqdm(folder_list):
        now_base_path = base_path + '%s/'%(now_folder)
        now_out_base_path = out_base_path + '%s/'%(now_folder)
        if not os.path.exists(now_out_base_path):
            os.makedirs(now_out_base_path)
        file_list = glob(now_base_path + '*.jpg')

        batch_size = 1
        right = 0
        wrong = 0
        with torch.no_grad():
            
            # now_input =  input('batch_size : ')
            # now_input = base_path + now_input
            for now_path in tqdm(file_list):
                
                input_list = [now_path]
                origin_img = cv2.imread(now_path)
                # h, w, c = origin_img.shape 
                # w, h = get_wh(w, h)
                # w = FloatTensor([w])
                # h = FloatTensor([h])
                if origin_img is None:
                    print(now_path , ' is None!')
                    continue
                # now_cls = get_cls_by_path(now_path)

                    
                # ta = time.time()
                # t = image_loader(data_transforms, input_list, device, batch_size)
                instance_img = origin_img
                instance_input = data_transforms(instance_img).unsqueeze(0)
                # tb = time.time()
                output = model_p(instance_input)
                prob = output[0].softmax(0)
                output = output.cpu().detach().numpy()
                # tb = time.time()
                # print('load data use time : %.2f'%(tb - ta))
                # exit()
                # output, _ = model_p(t, w, h)
                # output = output.cpu().detach().numpy()
                # output = model_p(t).cpu().detach().numpy()
                # print(output.shape)
                for j in range(output.shape[0]):
                    pred = np.argmax(output[j])
                    if pred in id_name_map.keys():
                        now_cls = id_name_map[pred]
                    else:
                        now_cls = pred
                    now_save_path = now_out_base_path + '%d/'%(int(now_cls))
                    if not os.path.exists(now_save_path):
                        os.makedirs(now_save_path)
                    save_id = len(os.listdir(now_save_path))
                    cv2.imwrite('%s%d_%d_%d.jpg'%(now_save_path, int(now_cls), save_id, it), origin_img)
                    it += 1

                


