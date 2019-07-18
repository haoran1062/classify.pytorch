# encoding:utf-8
import numpy as np, os, sys, shutil, cv2, torch
from backbones.DenseNet_features import densenet121

import torch.nn as nn
from sklearn.cluster import KMeans  
from torchsummary import summary
from torchvision import transforms, utils
from glob import glob
from tqdm import tqdm

def process_one_folder(now_base_in_path, now_base_out_path, model_p, cls_n, it):

    

    tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    all_t = []
    path_l = []
    img_list = glob(now_base_in_path + '*.jpg')
    cls_n = min(len(img_list), cls_n)
    kmeans=KMeans(n_clusters=cls_n)

    for now_img_path in tqdm(img_list):
        # print(now_img_path)
        in_img = cv2.imread(now_img_path)
        if in_img is None:
            print(now_img_path)
            
        in_img = cv2.resize(in_img, (224, 224))
        t_img = transforms.ToTensor()(in_img)
        t_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t_img)
        t_img.unsqueeze_(0)
        t_img = t_img.to(device)

        t = model.forward(t_img)
        t = t.view(1, -1)

        all_t.append(t.cpu().detach())
    

    total_t = torch.cat(all_t, 0)
    # print(total_t.shape)
    x = total_t.cpu().detach().numpy()
    kmeans.fit(x)
    # print(kmeans.labels_)
    ans = kmeans.labels_.tolist()


    ii = 0
    for i in tqdm(ans):
        now_save_path = now_base_out_path + '%d/'%(i)
        if not os.path.exists(now_save_path):
            os.makedirs(now_save_path)
        now_in_path = img_list[ii]
        now_out_path = now_save_path + os.path.split(now_in_path)[-1]
        shutil.copyfile(now_in_path, now_out_path)
        ii += 1

        it += 1
    return it

if __name__ == "__main__":
    # base_in_path = '/data/results/temp/classify_instance/201907/0703_night_instances/'
    # base_out_path = '/data/results/temp/classify_instance/201907/cls_out/0703_classify/night/'

    base_in_path = '/data/results/temp/classify_instance/201907/0717_1_instances/'
    base_out_path = '/data/results/temp/classify_instance/201907/cls_out/0717_1_pre_cls/'

    # base_in_path = '/data/results/temp/classify_instance/201907/0703_7_instances/'
    # base_out_path = '/data/results/temp/classify_instance/201907/cls_out/0703_7_pre_cls/'
    if not os.path.exists(base_out_path):
        os.makedirs(base_out_path)

    device = 'cuda:0'
    cuda = True
    cls_n = 2
    
    
    cnn_resumt_path = '/home/ubuntu/project/classify.pytorch/saved_models/densenet121_top100_224_truth_and_sync_with_wh/best.pth'
    model = densenet121(pretrained=False)
    model_p = nn.DataParallel(model.to(device), device_ids=[0])
    model_p.load_state_dict(torch.load(cnn_resumt_path), strict=False)
    model_p.eval()

    it = 0
    folder_list = os.listdir(base_in_path)
    for now_folder in tqdm(folder_list[:]):
        now_base_in_path = base_in_path + '%s/'%(now_folder)
        now_base_out_path = base_out_path + '%s/'%(now_folder)
        it = process_one_folder(now_base_in_path, now_base_out_path, model_p, cls_n, it)
        

    

    
