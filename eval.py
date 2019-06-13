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

cfg_path = 'configs/classify2050c_densenet121_eval.json'
cfg_path = 'configs/classify100c_densenet121_eval.json'
config_map = get_config_map(cfg_path)

# model_name = 'new50c_densenet_gan_data'
# model_name = 'new50c_densenet_only_sync'
# model_name = 'new50c_densenet_mix_data'
# model_name = 'new50c_densenet_only_truth' # _test_on_train'
# model_path = '/data/projects/classify_pytorch/save_weights/densenet/new50c_densenet_gan_data/densenet/epoch_6_step_0_acc_0.94.pth'
# model_path = '/data/projects/classify_pytorch/save_weights/densenet/%s/densenet/best_model.pth'%(model_name)
# model_path = '/data/projects/classify_pytorch/save_weights/densenet/new50c_densenet_only_truth/densenet/epoch_8_step_0_acc_0.95.pth'
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

data_transforms = transforms.Compose([
            transforms.Lambda(lambda img: padding_resize(img)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

def get_iter(now_num):
    if now_num > 47:
        now_num %= len(str_l)
        print(now_num, 'error!!!')
    return str_l[now_num]
    
def get_cls_by_path(in_path):
    sl = in_path.split('/')
    return int(sl[-2])

def write_result(out_path, right, wrong):
    with open(out_path, 'w') as f:
        f.write('right: %d\nwrong: %d\ntotal: %d\nacc: %.3f'%(right, wrong, right + wrong, float(right)/(right + wrong)))

if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model_ft, input_size = load_model(model_path)
    model_ft, input_size = initialize_model(config_map['model_type'], config_map['class_number'], config_map['feature_extract'], use_pretrained=False)

    model_p = nn.DataParallel(model_ft.to(device), device_ids=config_map['gpu_ids'])
    if config_map['load_from_path']:
        model_p.load_state_dict(torch.load(config_map['load_from_path']))
    model_p.eval()

    
    # print(model_ft)
    # base_path = '/data/datasets/train_data/new50c_classify_data/gan_data/test/'
    base_path = '/data/datasets/truth_data/classify_data/top100_checkout/train/origin/top100_multi_truth_train_20190612/'
    # base_path = '/data/datasets/train_data/new50c_classify_data/only_truth/train/'
    
    # out_base_path = '/data/projects/classify_pytorch/results/vis_results/%s/'%(model_name)
    # badcase_path = out_base_path + 'badcase/'
    # right_path = out_base_path + 'right/'

    file_list = glob(base_path + '*/*.jpg')

    batch_size = 1
    right = 0
    wrong = 0
    with torch.no_grad():
        
        # now_input =  input('batch_size : ')
        # now_input = base_path + now_input
        bar = tqdm(total=len(file_list))        
        for it, now_path in enumerate(file_list):
            
            input_list = [now_path]
            origin_img = cv2.imread(now_path)
            now_cls = get_cls_by_path(now_path)

                
            # ta = time.time()
            t = image_loader(data_transforms, input_list, device, batch_size)
            # tb = time.time()
            # print('load data use time : %.2f'%(tb - ta))
            # exit()
            output = model_p(t).cpu().detach().numpy()
            # print(output.shape)
            for j in range(output.shape[0]):
                pred = np.argmax(output[j])
                print(pred)
                # ans = int(get_iter(pred))
                ans = pred
                if ans == now_cls:
                    print('right!')
                    # if not os.path.exists(right_path + '%d'%(now_cls)):
                    #     os.makedirs(right_path + '%d'%(now_cls))
                    # cv2.imwrite(right_path + '%d/%d_%d.jpg'%(now_cls, now_cls, it), origin_img)
                    right += 1
                else:
                    print('wrong!')
                    # if not os.path.exists(badcase_path + '%d'%(now_cls)):
                    #     os.makedirs(badcase_path + '%d'%(now_cls))
                    # cv2.imwrite(badcase_path + '%d/%d---%d_%d.jpg'%(now_cls, now_cls, ans, it), origin_img)
                    wrong += 1
            
            bar.update(1)
        bar.close()
        # write_result(out_base_path+'result.txt', right, wrong)
                # print( get_iter(pred) )

                # tc = time.time()
                # print('predict use time : %.2f'%(tc - tb))

            
            # now_input = input('batch_size : ')

