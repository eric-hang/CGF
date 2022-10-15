import os
import time
import pprint
import torch
import numpy as np
import scipy.sparse as sp
import tcvae.tcvae as tcvae
from tcvae.tcvae import logsumexp
import math
import torch.nn.functional as F
def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def one_hot(indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def MGD_estimate(query,prototypes,support_labels_one_hot,support,n_way,logits=None):
    scale = 10
    if logits is None:
        logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    assign = F.softmax(logits * scale, dim=-1)
    assign = torch.cat([support_labels_one_hot, assign], dim=1)
    assign_transposed = assign.transpose(1, 2)
    emb = torch.cat([support, query], dim=1)
    mean = torch.bmm(assign_transposed, emb)
    mean = mean.div(
        assign_transposed.sum(dim=2, keepdim=True).expand_as(mean)
    )
    diff = torch.pow(emb.unsqueeze(1).expand(-1, n_way, -1, -1) - mean.unsqueeze(2).expand(-1, -1, emb.shape[1], -1), 2)
    std = (assign_transposed.unsqueeze(-1).expand_as(diff) * diff).sum(dim=2) / assign_transposed.unsqueeze(-1).expand_as(diff).sum(dim=2)
    return logits,mean,std
def FuseCosineNetHead(query, support, support_labels, n_way, n_shot,boost_prototypes,query_gen = None,normalize_=True,is_prototype=False):
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)


    labels_train_transposed = support_labels_one_hot.transpose(1, 2)

    prototypes = torch.bmm(labels_train_transposed, support)
    
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    boost_prototypes = boost_prototypes.reshape(tasks_per_batch,n_way,n_support,-1)[:,:,:10,:].mean(dim=2)

    counterfactual_logits = torch.nn.functional.cosine_similarity(torch.repeat_interleave(query.squeeze(),n_way,0).reshape(-1,640),query_gen, dim=-1).reshape(tasks_per_batch,n_query,-1)

    if normalize_:
        boost_prototypes = F.normalize(boost_prototypes)
        query = F.normalize(query)


    original_logits,mean_1,std_1 = MGD_estimate(query,prototypes,support_labels_one_hot,support,n_way)

    completed_logits,mean_2,std_2 = MGD_estimate(query,boost_prototypes,support_labels_one_hot,support,n_way)

    counterfactual_logits,mean_3,std_3 = MGD_estimate(query,None,support_labels_one_hot,support,n_way,logits=counterfactual_logits)


    prototypes_gen = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)
    prototypes_counterfactual = (mean_1 * std_3 + mean_3 * std_1) / (std_3 + std_1)


    _,mean_1,std_1 = MGD_estimate(query,0.2 * prototypes_gen + (1-0.2) * prototypes_counterfactual,support_labels_one_hot,support,n_way)
    _,mean_2,std_2 = MGD_estimate(query,prototypes_counterfactual,support_labels_one_hot,support,n_way)

    prototypes = (mean_1 * std_2 + mean_2 * std_1) / (std_2 + std_1)

    logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    if is_prototype:
        return prototypes,prototypes_gen,prototypes_counterfactual
    else:
        return logits,original_logits,completed_logits

def FuseCosineNetHead_base(query, support, support_labels, n_way, n_shot,normalize_=True):
    scale = 10
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)


    labels_train_transposed = support_labels_one_hot.transpose(1, 2)

    
    prototypes = torch.bmm(labels_train_transposed, support)
    
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )
    
    if normalize_:
        boost_prototypes = F.normalize(boost_prototypes)
        query = F.normalize(query)

    original_logits = torch.nn.functional.cosine_similarity(query.unsqueeze(2).expand(-1, -1, prototypes.shape[1], -1),
                                                   prototypes.unsqueeze(1).expand(-1, query.shape[1], -1, -1), dim=-1)
    
    return original_logits

from sklearn.linear_model import LogisticRegression
def clsHead(query, support, support_labels, n_way, n_shot,x_gen,gen_label):

    assert (support.shape[0] == n_way * n_shot)  # n_support must equal to n_way * n_shot
    assert (x_gen.shape[0] == n_way * n_shot * n_way)  # n_support must equal to n_way * n_shot

    query = F.normalize(query)
    support = F.normalize(support)
    x_gen = F.normalize(x_gen)
    clf = LogisticRegression(penalty='l2',random_state=0, C=1.0,solver='lbfgs',max_iter=1000,multi_class='multinomial')

    support_features = support.detach().cpu().numpy()
    support_ys = support_labels.detach().cpu().reshape(-1).numpy()

    clf.fit(support_features, support_ys)
    original_logits = clf.predict(query.detach().cpu().numpy())

    clf = LogisticRegression(penalty='l2',random_state=0, C=1.0,solver='lbfgs',max_iter=1000,multi_class='multinomial')

    support_features = torch.cat([support,x_gen],0).detach().cpu().numpy()
    support_ys = torch.cat([support_labels.reshape(-1),gen_label],0).detach().cpu().numpy()

    clf.fit(support_features, support_ys)
    logits = clf.predict(query.detach().cpu().numpy())
    
    clf = LogisticRegression(penalty='l2',random_state=0, C=1.0,solver='lbfgs',max_iter=1000,multi_class='multinomial')

    support_features = x_gen.detach().cpu().numpy()
    support_ys = gen_label.detach().cpu().reshape(-1).numpy()

    clf.fit(support_features, support_ys)
    completed_logits = clf.predict(query.detach().cpu().numpy())

    return logits,original_logits,completed_logits


def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy

def get_tcvae_kl_loss(mean, log_var, z, beta, opt):
    batch_size = mean.shape[0]
    z_dim = mean.shape[1]
    log_var *= 0.5
    prior_parameters = torch.zeros(batch_size, z_dim, 2).cuda()
    z_params = torch.cat((mean.unsqueeze(2), log_var.unsqueeze(2)), dim=2)
    prior_dist = tcvae.Normal()
    q_dist = tcvae.Normal()
    zs = z
    logpz = prior_dist.log_density(zs, params=prior_parameters).view(batch_size, -1).sum(1)
    logqz_condx = q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
    _logqz = q_dist.log_density(
        zs.view(batch_size, 1, z_dim),
        z_params.view(1, batch_size, z_dim, q_dist.nparams)
    )
    logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * opt.ntrain)).sum(1)
    logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * opt.ntrain))
    kld = (logqz_condx - logqz) + beta * (logqz - logqz_prodmarginals) + (logqz_prodmarginals - logpz)
    return kld.mean()

def loss_fn(recon_x, x, mean, log_var, opt, z, beta=1.0):
    if opt.recon == "bce":
        BCE = torch.nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='sum')
        BCE = BCE.sum() / x.size(0)
    elif opt.recon == "l2":
        BCE = torch.sum(torch.pow(recon_x - x.detach(), 2), 1).mean()
    elif opt.recon == "l1":
        BCE = torch.sum(torch.abs(recon_x - x.detach()), 1).mean()
    if not opt.z_disentangle:
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    elif not opt.zd_tcvae:
        KLD = -0.5 * beta * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    else:
        KLD = get_tcvae_kl_loss(mean, log_var, z, beta, opt)

    return (BCE + KLD)


def conditional_sample(netE,  netG, x, y):

    means, log_var = netE(F.normalize(x))
    z = sample_with_gradient(means, 0.5 * log_var)

    x_gen = netG(z, c=y)
    return x_gen


def sample_with_gradient(mean, log_var):
    batch_size = mean.shape[0]
    latent_size = mean.shape[1]
    std = torch.exp(log_var)
    eps = torch.randn([batch_size, latent_size]).cpu()
    eps = Variable(eps.cuda())
    z = eps * std + mean
    return z

def construct_negative_class(opt,K):
    if opt.dataset == 'miniImageNet' or opt.dataset == 'CIFAR-FS':
        train_classes = [_ for _ in range(64)]
    elif opt.dataset == 'tieredImageNet':
        train_classes = [_ for _ in range(351)]

    neg_classes_dict = dict()
    for i in range(len(train_classes)):
        negative_classes = [item for item in train_classes if item != train_classes[i]]
        negative_classes_selected = random.sample(negative_classes, K)
        neg_classes_dict[train_classes[i]] = negative_classes_selected
    return neg_classes_dict

    
from torch.autograd import Variable
import random
def contrastive_loss(netE, netG, batch_x, batch_l, attribute_dict,pre_head,opt,neg_classes_dict,temperature=1.0, K=128,class_feature = None):

    batch_size = batch_x.shape[0]
    x_dim = batch_x.shape[1]
    s_dim = opt.attSize
    batch_s = attribute_dict[batch_l]

    attributes = Variable(torch.zeros((batch_size, K + 1, s_dim))).cuda()
    x = Variable(torch.zeros((batch_size, K + 1, x_dim))).cuda()
    negative_label = Variable(torch.zeros((batch_size, K + 1))).long().cuda()

    for i in range(batch_size):
        # First element is ground-truth
        attributes[i][0] = batch_s[i]
        # Negative samples
        attributes[i, 1:] = attribute_dict[neg_classes_dict[batch_l[i].item()], :]
        x[i] = batch_x[i].unsqueeze(0).expand(K + 1, -1)
        negative_label[i,:] = torch.tensor([batch_l[i]] + neg_classes_dict[batch_l[i].item()])

    xv = x.view(batch_size * (K + 1), -1)
    yv = attributes.view(batch_size * (K + 1), -1)

    dictionary = conditional_sample(netE,  netG,  xv, yv)
    ori_prototype = class_feature[negative_label.view(batch_size * (K + 1), -1).cpu().squeeze().tolist(),:]

    dictionary = dictionary.view(batch_size, K + 1, x_dim)
    if opt.network == 'ResNet12':
        logits = pre_head(dictionary)
    else:
        logits = pre_head.predict(dictionary)
    classify_loss = torch.nn.CrossEntropyLoss()(logits.reshape(-1,len(neg_classes_dict.keys())),negative_label.reshape(-1))
    batch_x_expanded = batch_x.unsqueeze(1).expand(batch_size, K + 1, -1)

    neg_dist = -((batch_x_expanded - dictionary) ** 2).mean(dim=2) * temperature

    label = torch.zeros(batch_size).cuda().long()

    contrastive_loss_euclidean = torch.nn.CrossEntropyLoss()(neg_dist, label)

    return contrastive_loss_euclidean + classify_loss
class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)