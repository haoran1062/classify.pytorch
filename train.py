#encoding:utf-8
import os, sys, time, numpy as np, cv2, copy, argparse
from glob import glob 

import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from utils.data_utils import *
from utils.model_utils import *
from utils.train_utils import *
from utils.visual import Visual
from torchsummary import summary
from utils.ClassifyDataLoader import ClassifyDataset

parser = argparse.ArgumentParser(
    description='Classify Training params')
# parser.add_argument('--config', default='configs/classify100c_densenet121.json')
parser.add_argument('--config', default='configs/classify800c_se-resnext50_512.json')

args = parser.parse_args()

config_map = get_config_map(args.config)

if not os.path.exists(config_map['model_save_path']):
    os.makedirs(config_map['model_save_path'])

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, epoch_start=0, save_base_path='./', save_step=500, logger=None, vis=None, rename_map=None, id_name_map=None):
    since = time.time()
    
    # cosin_lr = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs // 10)+1)
    adjust_lr = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.8, verbose=1, patience=2)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epoch_start, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        acc_map = {}
        # cosin_lr.step(epoch)
        my_vis.plot('lr', optimizer.param_groups[0]['lr'])

        # Each epoch has a training and validation phase
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for it, temp in enumerate(dataloaders[phase]):
                inputs, labels = temp
                inputs = inputs.to(device)
                labels = labels.to(device)
                st = time.clock()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                now_loss = loss.item() * inputs.size(0)
                running_loss += now_loss
                now_correct = torch.sum(preds == labels.data)
                running_corrects += now_correct

                if phase == 'test':
                    
                    p_l = preds.data.tolist()
                    gt_l = labels.data.tolist()
                    for tij in range(len(p_l)):
                        t_gt = gt_l[tij]
                        if t_gt not in acc_map.keys():
                            acc_map[t_gt] = [0, 0]
                        if t_gt == p_l[tij]:
                            acc_map[t_gt][0] += 1
                        else:
                            acc_map[t_gt][1] += 1


                ed = time.clock()
                it_cost_time = ed - st
                
                if it % 10 == 0:
                    # convert_show_cls_bar_data(acc_map, rename_map=rename_map)
                    if phase == 'train':
                        logger.info('Epoch [{}/{}], Iter [{}/{}] expect end in {:4f} min.  average_loss: {:2f}, now Acc: {}'.format(epoch, config_map['epoch_number'], it, len(dataloaders[phase]), it_cost_time * (len(dataloaders[phase]) - it+1) / 60, running_loss / (it+1), running_corrects.double()/((it+1) * inputs.shape[0]) ) )

                    img = tensor2img(inputs, normal=True)
                    vis.img('show img', img)
                    if id_name_map:
                        show_id = preds.to('cpu').numpy()[0]
                        if show_id in id_name_map.keys():
                            show_id = id_name_map[show_id]
                        vis.img('pred result', get_show_result_img(id_name_map[labels.to('cpu').numpy()[0]], show_id))
                    else:
                        vis.img('pred result', get_show_result_img(labels.to('cpu').numpy()[0], preds.to('cpu').numpy()[0]))

                if it % save_step == 0 and phase == 'train':
                    if not os.path.exists('%s'%(save_base_path)):
                        os.mkdir('%s'%(save_base_path))
                    torch.save(model.state_dict(), '%s/epoch_%d.pth'%(save_base_path, epoch))
                    

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            adjust_lr.step(epoch_acc)

            if phase == 'train':
                my_vis.plot('train loss', epoch_loss)
                my_vis.plot('train acc', epoch_acc.item())

            elif phase == 'test':
                my_vis.plot('test loss', epoch_loss)
                my_vis.plot('test acc', epoch_acc.item())

                acc_x, leg_l, name_l = convert_show_cls_bar_data(acc_map, rename_map=rename_map)
                my_vis.multi_cls_bar('every class Acc', acc_x, leg_l, name_l)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                pass

    time_elapsed = time.time() - since
    logger.info('finish training using %.2fs'%(time_elapsed))
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), '%s/best.pth'%(save_base_path))



if __name__ == "__main__":

    fine_turn = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    img_input_size = config_map['input_size']
    
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.Lambda(lambda img: origin_resize(img)),
            transforms.Lambda(lambda img: padding_resize(img, resize=img_input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.Lambda(lambda img: origin_resize(img)),
            transforms.Lambda(lambda img: padding_resize(img, resize=img_input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(input_size),
            # transforms.Lambda(lambda img: origin_resize(img)),
            transforms.Lambda(lambda img: padding_resize(img, resize=img_input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Initialize the model for this run
    model_ft, input_size = initialize_model(config_map['model_type'], config_map['class_number'], config_map['feature_extract'], use_pretrained=True)
    
    # model_ft.load_state_dict(torch.load(config_map['resume_from_path']))
    model_p = nn.DataParallel(model_ft.to(device), device_ids=config_map['gpu_ids'])
    if config_map['resume_from_path']:
        print("resume from %s"%(config_map['resume_from_path']))
        model_p.load_state_dict(torch.load(config_map['resume_from_path']))
        
   
    # Print the model we just instantiated
    # summary(model_p, (3, img_input_size, img_input_size))

    # Send the model to GPU

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.

    logger = create_logger(config_map['model_save_path'], config_map['log_name'])

    my_vis = Visual(config_map['model_save_path'], log_to_file=config_map['vis_log_path'])   

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_p.parameters(), lr=config_map['batch_size'] / 256.0, momentum=0.9)
    # optimizer_ft = optim.RMSprop(params_to_update, momentum=0.9)
    # optimizer_ft = optim.Adam(model_p.parameters(), lr=1e-2, eps=1e-8, betas=(0.9, 0.99), weight_decay=0.)
    # optimizer_ft = optim.Adadelta(params_to_update, lr=1)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    dataloaders = {}
    train_dataset = ClassifyDataset(base_data_path=config_map['train_data_path'], train=True, transform = data_transforms['train'], id_name_path=config_map['id_name_txt'], device=device, little_train=False)
    train_loader = DataLoader(train_dataset,batch_size=config_map['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = ClassifyDataset(base_data_path=config_map['test_data_path'], train=False,transform = data_transforms['val'], id_name_path=config_map['id_name_txt'], device=device, little_train=False, with_file_path=False)
    test_loader = DataLoader(test_dataset,batch_size=config_map['batch_size'],shuffle=True, num_workers=4, pin_memory=True)
    id_name_map = train_dataset.id_name_map
    data_len = int(len(test_dataset) / config_map['batch_size'])
    logger.info('the dataset has %d images' % (len(train_dataset)))
    logger.info('the batch_size is %d' % (config_map['batch_size']))

    dataloaders['train']=train_loader
    dataloaders['test']=test_loader

    model_p.train()
    # Train and evaluate
    train_model(model_p, dataloaders, criterion, optimizer_ft, num_epochs=config_map['epoch_number'], epoch_start=config_map['resume_epoch'], save_base_path=config_map['model_save_path'], logger=logger, vis=my_vis, rename_map=id_name_map, id_name_map=id_name_map)

