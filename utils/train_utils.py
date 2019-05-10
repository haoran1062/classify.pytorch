# encoding:utf-8
import os, numpy as np, random, cv2, logging, json



def get_config_map(file_path):
    config_map = json.loads(open(file_path).read())
    
    config_map['batch_size'] *= len(config_map['gpu_ids'])
    return config_map

def create_logger(base_path, log_name):

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    fhander = logging.FileHandler('%s/%s.log'%(base_path, log_name))
    fhander.setLevel(logging.INFO)

    shander = logging.StreamHandler()
    shander.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    fhander.setFormatter(formatter) 
    shander.setFormatter(formatter) 

    logger.addHandler(fhander)
    logger.addHandler(shander)

    return logger

def get_show_result_img(gt_label, pred_label):
    img = np.zeros((100, 500, 3), np.uint8)
    str_input = 'gt: %d, pred : %d'%(gt_label, pred_label)
    cv2.putText(img, str_input, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
    return img