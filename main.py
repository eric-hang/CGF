# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch

from config import opt
from get_model_data import get_dataset
from pretrain import pre_train
from meta_train import meta_train
from meta_test import meta_test
from pre_train_encoder_vaegan import pre_train_encoder_vaegan
from pre_train_encoder_vae import pre_train_encoder_vae
from pre_train_encoder_visdec import pre_train_encoder_visdec
from data_scaler import data_scaler
from save_part import part_prototype
from test import test
from proto_complete import meta_inference
from test_base import test_base
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)
def seed_torch(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_gpu(opt.gpu)
    seed_torch(opt.seed)
    
    (dataset_train, dataset_val, dataset_test, data_loader) = get_dataset(opt)

    if opt.phase == 'pretrain':
        pre_train(opt, dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'metatrain':
        meta_train(opt, dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'metatest':
        meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'savepart':
        part_prototype(opt, dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'pretrain_encoder':
        print('pretrain_encoder')
        if opt.generative_model == 'vaegan':
            pre_train_encoder_vaegan(opt,dataset_train, dataset_val, dataset_test, data_loader)
        elif opt.generative_model == 'vae':
            pre_train_encoder_vae(opt,dataset_train, dataset_val, dataset_test, data_loader)
        else:
            pre_train_encoder_visdec(opt,dataset_train, dataset_val, dataset_test, data_loader)

    elif opt.phase == 'data_scaler':
        data_scaler(opt, dataset_train, dataset_val, dataset_test, data_loader)
    elif opt.phase == 'generative_test':
        test(opt,dataset_test,data_loader)
    elif opt.phase == 'generative_test_base':
        test_base(opt,dataset_test,data_loader)
    else:
        meta_inference(opt, dataset_train, dataset_val, dataset_test, data_loader)