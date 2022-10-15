import torch
import pickle
import os
from utils import contrastive_loss, set_gpu, Timer, count_accuracy, check_dir, log, loss_fn,construct_negative_class
from get_model_data import get_model,get_dataset
from tqdm import tqdm
from tcvae.tcvae import anneal_kl
from torch.autograd import Variable
import numpy as np
import joblib
import torch.nn.functional as F
from valid import valid
import models.resnet_torch as resnet_torch
def pre_train_encoder_vae(opt, dataset_train, dataset_val, dataset_test, data_loader):
    data_loader_pre = torch.utils.data.DataLoader
    if opt.use_trainval == 'True':
        train_way = 80
        dloader_train = data_loader_pre(
            dataset=dataset_trainval,
            batch_size=128,
            shuffle=True,
            num_workers=0
        )
    else:
        train_way = 64
        dloader_train = data_loader_pre(
            dataset=dataset_train,
            batch_size=128,
            shuffle=True,
            num_workers=0
        )
    print(len(dloader_train))
    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "pretrain_encoder_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, netE,netG,_,_,_,_, pre_head, cls_head) = get_model(opt)

    if opt.network == 'ResNet12':
        saved_models = torch.load('./experiments/resnet12/best_pretrain_model.pth')
        embedding_net.load_state_dict(saved_models['embedding'])
        embedding_net.eval()
        pre_head.load_state_dict(saved_models['pre_head'])
        pre_head.eval()
    elif opt.network == 'ResNet12_pretrain':
        if opt.dataset == 'miniImageNet':
            saved_models = torch.load('./pretrained/mini_distilled.pth')
            embedding_net.load_state_dict(saved_models['model'])
            embedding_net.eval()
        elif opt.dataset == 'tieredImageNet':
            saved_models = torch.load('./pretrained/resnet12_last.pth')
            embedding_net.load_state_dict(saved_models['model'])
            embedding_net.eval()
        elif opt.dataset == 'CIFAR-FS':
            saved_models = torch.load('./pretrained/fs_distill_correct.pth')
            embedding_net.load_state_dict(saved_models['model'])
            embedding_net.eval()
        else:
            assert False
    elif opt.network == 'ResNet12_inv':
        if opt.dataset == 'tieredImageNet':
            embedding_net = torch.nn.DataParallel(embedding_net)
            saved_models = torch.load('./pretrained/tiered_inv.pth')
            embedding_net.load_state_dict(saved_models['model'])
            embedding_net.eval()
            embedding_net = embedding_net.module
        elif opt.dataset == 'miniImageNet':
            embedding_net = torch.nn.DataParallel(embedding_net)
            saved_models = torch.load('./pretrained/mini_inv.pth')
            embedding_net.load_state_dict(saved_models['model'])
            embedding_net.eval()
            embedding_net = embedding_net.module
        elif opt.dataset == 'CIFAR-FS':
            saved_models = torch.load('./pretrained/fs_inv.pth')
            embedding_net.load_state_dict(saved_models['model'])
            embedding_net.eval()
    else:
        assert False

    if opt.baseline is False:
        if opt.dataset == 'miniImageNet':
            matcontent = np.load('./dataset/MiniImagenet/miniImageNet.npz',allow_pickle=True)
            attribute_dict = torch.from_numpy(matcontent['attribute']).float().squeeze(1)
            attribute_dict /= attribute_dict.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute_dict.size(0),attribute_dict.size(1))
            with open('./experiments/mini_feature/mini_feat_inv.pickle', 'rb') as handle:
                class_feature = pickle.load(handle)
            class_feature = class_feature.reshape(64,600,-1).mean(dim=1)
        elif opt.dataset == 'tieredImageNet':

            matcontent = np.load('./dataset/tieredImagenet/tieredImagenet.npz',allow_pickle=True)
            attribute_dict = torch.from_numpy(matcontent['attribute']).float().squeeze(1)

            print(attribute_dict.shape)
            attribute_dict /= attribute_dict.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute_dict.size(0),attribute_dict.size(1))
            with open('./experiments/tie_feature/tie_class_prototype_inv.pickle', 'rb') as handle:
                class_feature = pickle.load(handle)
            class_feature = torch.cat(class_feature,dim=0).cuda()
            
        elif opt.dataset == 'CIFAR-FS':
            matcontent = np.load('./dataset/cifar-fs/cifar-fs.npz',allow_pickle=True)
            attribute_dict = torch.from_numpy(matcontent['attribute']).float().squeeze(1)
            attribute_dict /= attribute_dict.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute_dict.size(0),attribute_dict.size(1))
            with open('./experiments/fs_feature/fs_feat_inv.pickle', 'rb') as handle:
                class_feature = pickle.load(handle)
            class_feature = class_feature.reshape(64,600,-1).mean(dim=1)
    else:
        attribute_dict = torch.zeros([100,171])
    attribute_dict = attribute_dict.cuda()

    origin_prototype = class_feature


    optimizer = torch.optim.Adam([{'params': netE.parameters()},
                                 {'params': netG.parameters()}],lr=opt.lr,betas=(opt.beta1, 0.999),weight_decay=5e-4)

    max_val_acc = 0.0

    if opt.dataset == 'tieredImageNet':
        K = 128
    else:
        K = 30
    neg_classes_dict = construct_negative_class(opt,K)

    for epoch in range(1, opt.num_epoch + 1):
        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {}'.format(
            epoch, epoch_learning_rate))

        _, _, _ = [x.eval() for x in (embedding_net, pre_head, cls_head)]
        _, _= [x.train() for x in (netE,netG)]
        train_accuracies = []
        train_losses = []
        beta = anneal_kl(opt, epoch)

        import time
        for i, batch in enumerate(tqdm(dloader_train)):

            data, labels = [x.cuda() for x in batch]
            nb, ns, nc, nw, nh = data.shape
            if opt.pre_head == 'LinearNet' or opt.pre_head == 'CosineNet':
                data = data.reshape(nb*ns, nc, nw, nh)
                with torch.no_grad():
                    emb = embedding_net(data)

                means, log_var = netE(F.normalize(emb))
                std = torch.exp(0.5 * log_var)
                eps = torch.randn(std.shape,device='cuda:0')
                z = eps * std + means

                additional_input_feature= attribute_dict[torch.repeat_interleave(labels,opt.train_shot).detach().cpu().tolist()].float()

                recon_x = netG(z,c=additional_input_feature)

                input_eps = torch.randn(emb.shape,device='cuda:0')
                hard_input = emb + opt.add_noise * input_eps

                vae_loss = loss_fn(F.normalize(recon_x),F.normalize(hard_input),means,log_var,opt,z,beta=beta)
                if opt.baseline == False:
                    if opt.network == 'ResNet12':
                        contrastive_loss_ = contrastive_loss(netE, netG,emb,torch.repeat_interleave(labels,opt.train_shot).detach(),attribute_dict,pre_head,opt,neg_classes_dict = neg_classes_dict,K=K,temperature=opt.temperature,class_feature=origin_prototype)
                    else:
                        contrastive_loss_ = contrastive_loss(netE, netG,emb,torch.repeat_interleave(labels,opt.train_shot).detach(),attribute_dict,embedding_net,opt,neg_classes_dict = neg_classes_dict,K=K,temperature=opt.temperature,class_feature=origin_prototype)

            
            if opt.baseline == False:
                loss = vae_loss + contrastive_loss_
            else:
                loss = vae_loss
            train_losses.append(loss.item())


            if (i % (len(dloader_train)//5) == 0):
                neg_classes_dict = construct_negative_class(opt,K)
                train_loss_mean = np.mean(np.array(train_losses))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}]\tLoss: {:.3f}\tavg_loss: {:.3f}\t'.format(
                    epoch, i, loss.item(),train_loss_mean))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return_acc = valid(epoch, embedding_net,netE,netG,netE,netE,netE,netE,max_val_acc,opt,dloader_val,attribute_dict,log_file_path)
        if return_acc is not None:
            max_val_acc = return_acc