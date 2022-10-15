from copyreg import pickle
import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils import count_accuracy,FuseCosineNetHead,clsHead
import os
import torch.nn.functional as F
from get_model_data import get_model
import models.resnet_torch as resnet_torch
import selectivesearch
import torchvision
import pickle
from sklearn import metrics
def test(opt,dataset_test,data_loader):
    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    (embedding_net, netE,netG,netD,netDec,netF,propa_head, _, _) = get_model(opt)

    if opt.baseline == False:
        if opt.dataset =='miniImageNet':
            matcontent = np.load('./dataset/MiniImagenet/miniImageNet.npz',allow_pickle=True)
            attribute_dict = torch.from_numpy(matcontent['attribute']).float().squeeze(1)
            attribute_dict /= attribute_dict.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute_dict.size(0),attribute_dict.size(1))
            with open('./experiments/mini_feature/mini_testproto_inv.pickle','rb')as f:
                true_proto = pickle.load(f)
        elif opt.dataset == 'tieredImageNet':
            matcontent = np.load('./dataset/tieredImagenet/tieredImagenet.npz',allow_pickle=True)
            attribute_dict = torch.from_numpy(matcontent['attribute']).float().squeeze(1)
            attribute_dict /= attribute_dict.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute_dict.size(0),attribute_dict.size(1))
        elif opt.dataset == 'CIFAR-FS':
            matcontent = np.load('./dataset/cifar-fs/cifar-fs.npz',allow_pickle=True)
            attribute_dict = torch.from_numpy(matcontent['attribute']).float().squeeze(1)
            attribute_dict /= attribute_dict.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute_dict.size(0),attribute_dict.size(1))
    else:
        attribute_dict = torch.zeros([100,171])
    # Load saved model checkpoints

    saved_models = torch.load('./experiments/mini_vae_inv_nomse/best_pretrain_model_meta_infer_val_5w_1s_FuseCosNet_pretrain_encoder.pth')

    embedding_net.load_state_dict(saved_models['embedding_net'])
    embedding_net.eval()
    netE.load_state_dict(saved_models['netE'])
    netE.eval()
    netG.load_state_dict(saved_models['netG'])
    netG.eval()
    if 'scale' in saved_models.keys() and  saved_models['scale'] is not None:
        scale = saved_models['scale'].cuda()
    else:
        scale = None

    if opt.generative_model == 'vaegan':
        netDec.load_state_dict(saved_models['netDec'])
        netDec.eval()
        netF.load_state_dict(saved_models['netF'])
        netF.eval()

    x_entropy = torch.nn.CrossEntropyLoss()
    _, _, _ = [x.eval() for x in (embedding_net, netE,netG)]

    test_accuracies = []
    original_acc = []
    completed_acc = []
    test_losses = []
    proto_sim = []
    for i, batch in enumerate(tqdm(dloader_test(opt.seed)), 1):
        data_support, labels_support, \
        data_query, labels_query, k_all, _ = [
            x.cuda() for x in batch]

        test_n_support = opt.test_way * opt.val_shot
        test_n_query = opt.test_way * opt.val_query
        with torch.no_grad():
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            
            means, log_var = netE(F.normalize(emb_support))
            std = torch.exp(0.5 * log_var)
            eps = Variable(torch.randn(means.shape,device='cuda:0'))
            z = eps * std + means
            
            z = z.repeat((opt.test_way,1))
            additional_input_feature = attribute_dict[torch.repeat_interleave(k_all.squeeze(),test_n_support).detach().cpu().tolist()].float().cuda()
            
            x_gen = netG(z,c=additional_input_feature)


            means, log_var = netE(F.normalize(emb_query))
            std = torch.exp(0.5 * log_var)
            eps = Variable(torch.randn(means.shape,device='cuda:0'))
            z = eps * std + means
            
            z = torch.repeat_interleave(z,opt.test_way,0)
            additional_input_feature = attribute_dict[k_all.squeeze().repeat(test_n_query).detach().cpu().tolist()].float().cuda()

            query_gen = netG(z,c=additional_input_feature)
            


            if opt.generative_model=='vaegan':
                __ = netDec(x_gen)
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                x_gen = netG(z,a1=opt.a1, c=additional_input_feature, feedback_layers=feedback_out)
            
            if opt.head == 'FuseCosNet':
                emb_support = emb_support.reshape(1,test_n_support, -1)
                emb_query = emb_query.reshape(1, test_n_query, -1)
                prototypes,prototypes_gen,prototypes_counterfactual = FuseCosineNetHead(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot,x_gen,query_gen,normalize_=False,is_prototype = True)

                logits,original_logits,completed_logits = FuseCosineNetHead(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot,x_gen,query_gen,normalize_=False,is_prototype = False)
            elif opt.head == 'clsHead':
                logits,original_logits,completed_logits = clsHead(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot,x_gen,torch.repeat_interleave(torch.tensor([0,1,2,3,4]).long(),test_n_support).cuda())


            if scale is not None:
                logits = scale * logits
                original_logits = scale * original_logits
                completed_logits = scale * completed_logits

        loss = x_entropy(logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
        acc = count_accuracy(logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
        ori_acc = count_accuracy(original_logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
        com_acc = count_accuracy(completed_logits.reshape(-1, opt.test_way), labels_query.reshape(-1))

        test_accuracies.append(acc.item())
        original_acc.append(ori_acc.item())
        completed_acc.append(com_acc.item())

    test_acc_avg = np.mean(np.array(test_accuracies))
    completed_acc_avg = np.mean(np.array(completed_acc))
    original_acc_avg = np.mean(np.array(original_acc))
    test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.val_episode)

    test_loss_avg = np.mean(np.array(test_losses))

    print('Test Loss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)\t orignial_acc: {:.2f}\t completed_acc: {:.2f}' \
        .format(test_loss_avg, test_acc_avg, test_acc_ci95,original_acc_avg,completed_acc_avg))
    print(np.mean(np.array(proto_sim)))