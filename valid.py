import torch
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils import count_accuracy,FuseCosineNetHead,clsHead,log
import os
import torch.nn.functional as F
from datetime import *
from sklearn import metrics
def valid(epoch,embedding_net, netE,netG,netDec,netF,netD,propa_head,max_val_acc,opt,dloader_val,attribute_dict,file_path,scale = None):
    x_entropy = torch.nn.CrossEntropyLoss()
    _, _, _,_= [x.eval() for x in (embedding_net, netE,netG,propa_head)]
    if opt.generative_model == 'vaegan':
        netDec.eval()
        netF.eval()
    val_accuracies = []
    val_losses = []
    original_acc = []
    completed_acc = []
    gen_label = [_ for _ in range(opt.test_way)]
    gen_label = torch.repeat_interleave(torch.tensor(gen_label),opt.test_way * opt.val_shot).long().cuda()
    for i, batch in enumerate(tqdm(dloader_val(opt.seed)), 1):
        data_support, labels_support, \
        data_query, labels_query, k_all, _ = [
            x.cuda() for x in batch]

        test_n_support = opt.test_way * opt.val_shot
        test_n_query = opt.test_way * opt.val_query

        with torch.no_grad():
            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            
            means, log_var = netE(F.normalize(emb_support))
            std = torch.exp(0.5 * log_var)
            eps = Variable(torch.randn(means.shape,device='cuda:0'))
            z = eps * std + means
            
            z = z.repeat((opt.test_way,1))

            additional_input_feature = attribute_dict[torch.repeat_interleave(k_all.squeeze(),test_n_support).detach().cpu().tolist()].float().cuda()
                 
            x_gen = netG(z,c=additional_input_feature)
            
            if opt.generative_model =='vaegan':
                __ = netDec(F.normalize(x_gen))
                dec_hidden_feat = netDec.getLayersOutDet()
                feedback_out = netF(dec_hidden_feat)
                x_gen = netG(z,a1=opt.a1, c=additional_input_feature, feedback_layers=feedback_out)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))

            if opt.head == 'FuseCosNet':
                emb_support = emb_support.reshape(1,test_n_support, -1)

                means, log_var = netE(F.normalize(emb_query))
                std = torch.exp(0.5 * log_var)
                eps = Variable(torch.randn(means.shape,device='cuda:0'))
                z = eps * std + means
                
                z = torch.repeat_interleave(z,opt.test_way,0)
                additional_input_feature = attribute_dict[k_all.squeeze().repeat(test_n_query).detach().cpu().tolist()].float().cuda()
                query_gen = netG(z,c=additional_input_feature)
                emb_query = emb_query.reshape(1, test_n_query, -1)

                if opt.phase=='metainfer':
                    x_gen, _ = propa_head(x_gen.reshape(-1, emb_query.size(2)), k_all.reshape(-1),is_infer=True)
                logits,original_logits,completed_logits = FuseCosineNetHead(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot,x_gen,query_gen,normalize_=False)
            elif opt.head == 'cls':
                logits,original_logits,completed_logits = clsHead(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot,x_gen,gen_label)
            else:
                assert False
            
            if scale is not None:
                logits = scale * logits
                original_logits = scale * original_logits
                completed_logits = scale * completed_logits

            if opt.head == 'FuseCosNet':
                loss = x_entropy(logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
                acc = count_accuracy(logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
                ori_acc = count_accuracy(original_logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
                com_acc = count_accuracy(completed_logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
            else:
                loss = torch.tensor([0])
                acc = metrics.accuracy_score(labels_query.reshape(-1).detach().cpu().numpy(),logits) * 100
                ori_acc = metrics.accuracy_score(labels_query.reshape(-1).detach().cpu().numpy(),original_logits) * 100
                com_acc = metrics.accuracy_score(labels_query.reshape(-1).detach().cpu().numpy(),completed_logits) * 100

        val_accuracies.append(acc.item())
        completed_acc.append(com_acc.item())
        original_acc.append(ori_acc.item())
        val_losses.append(loss.item())

    val_acc_avg = np.mean(np.array(val_accuracies))
    original_acc_avg = np.mean(np.array(original_acc))
    completed_acc_avg = np.mean(np.array(completed_acc))
    val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

    val_loss_avg = np.mean(np.array(val_losses))
    if val_acc_avg > max_val_acc:
        max_val_acc = val_acc_avg
        torch.save({"netE":netE.state_dict(),"netG":netG.state_dict(),'embedding_net':embedding_net.state_dict(),'netF':netF.state_dict(),'netDec':netDec.state_dict(),'netD':netD.state_dict(),'propa_head': propa_head.state_dict(),'scale':scale}, \
                   os.path.join(opt.save_path, 'best_pretrain_model_meta_infer_val_{}w_{}s_{}_{}.pth'.format(opt.test_way, opt.val_shot, opt.head,opt.phase)))
        log(file_path,'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)\tcom_acc {:.2f}\tori_acc {:.2f}' \
            .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95,completed_acc_avg,original_acc_avg))
        return max_val_acc
    else:
        torch.save({"netE":netE.state_dict(),"netG":netG.state_dict(),'embedding_net':embedding_net.state_dict(),'netF':netF.state_dict(),'netDec':netDec.state_dict(),'netD':netD.state_dict(),'propa_head': propa_head.state_dict(),'scale':scale}, \
                   os.path.join(opt.save_path, 'latest_pretrain_model_meta_infer_val_{}w_{}s_{}_{}.pth'.format(opt.test_way, opt.val_shot, opt.head,opt.phase)))
        log(file_path,'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %\tcom_acc {:.2f}\tori_acc {:.2f}' \
            .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95,completed_acc_avg,original_acc_avg))
        return None