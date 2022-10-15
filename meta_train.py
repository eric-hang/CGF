from utils import set_gpu, Timer, count_accuracy, check_dir, log, loss_fn,FuseCosineNetHead
import os
from get_model_data import get_model,get_dataset
import torch
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from valid import valid
def meta_train(opt, dataset_train, dataset_val, dataset_test, data_loader):
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,  # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 100,  # num of batches per epoch
    )

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

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "meta_train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, netE,netG,_,_,_,_, pre_head, cls_head) = get_model(opt)

    if opt.baseline is False:
        if opt.dataset == 'miniImageNet':
            matcontent = np.load('./dataset/MiniImagenet/miniImageNet.npz',allow_pickle=True)
            attribute_dict = torch.from_numpy(matcontent['attribute']).float().squeeze(1)
            attribute_dict /= attribute_dict.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute_dict.size(0),attribute_dict.size(1))
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
    model_path = os.path.join(opt.save_path,'best_pretrain_model_meta_infer_val_5w_1s_FuseCosNet_pretrain_encoder.pth')
    saved_models = torch.load(model_path)


    embedding_net.load_state_dict(saved_models['embedding_net'])
    embedding_net.eval()
    netE.load_state_dict(saved_models['netE'])
    netE.eval()
    netG.load_state_dict(saved_models['netG'])
    netG.eval()
    embedding_net = torch.nn.DataParallel(embedding_net)
    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params':netE.parameters()},
                                 {'params':netG.parameters()}
                                 ], lr=0.0001, momentum=0.9, \
                                weight_decay=5e-4, nesterov=True)

    lambda_epoch = lambda e: 1.0 if e < 15 else (0.1 if e < 25 else 0.01 if e < 30 else (0.001))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

    max_val_acc = 0.0
    max_test_acc = 0.0

    x_entropy = torch.nn.CrossEntropyLoss()

    scale = torch.nn.Parameter(torch.FloatTensor([1.0])).cuda()

    for epoch in range(0, opt.num_epoch + 1):
        if epoch != 0:
            # Train on the training split
            # Fetch the current epoch's learning rate
            epoch_learning_rate = 0.1
            for param_group in optimizer.param_groups:
                epoch_learning_rate = param_group['lr']

            log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                epoch, epoch_learning_rate))

            _, _, _= [x.train() for x in (embedding_net, netE,netG)]

            train_accuracies = []
            original_acc = []
            completed_acc = []
            train_losses = []

            return_acc = valid(epoch, embedding_net,netE,netG,netE,netE,netE,netE,max_val_acc,opt,dloader_val,attribute_dict,log_file_path,scale)

            if return_acc is not None:
                max_val_acc = return_acc
            
            for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
                data_support, labels_support, \
                data_query, labels_query, k_all, _ = [
                    x.cuda() for x in batch]


                train_n_support = opt.train_way * opt.train_shot
                train_n_query = opt.train_way * opt.train_query
                emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
                emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
                means, log_var = netE(F.normalize(emb_support))
                std = torch.exp(0.5 * log_var)
                eps = Variable(torch.randn(means.shape,device='cuda:0'))

                z = eps * std + means
               
                z = z.repeat((5,1))
                additional_input_feature = attribute_dict[torch.repeat_interleave(k_all.squeeze(),train_n_support).detach().cpu().tolist()].float().cuda()

                x_gen = netG(z,c=additional_input_feature)


                means, log_var = netE(F.normalize(emb_query))
                std = torch.exp(0.5 * log_var)
                eps = Variable(torch.randn(means.shape,device='cuda:0'))
                z = eps * std + means
                
                z = torch.repeat_interleave(z,opt.test_way,0)

                additional_input_feature = attribute_dict[k_all.squeeze().repeat(train_n_query,1).detach().cpu().tolist()].float().cuda().reshape(-1,attribute_dict.shape[1])

                query_gen = netG(z,c=additional_input_feature)

                emb_support = emb_support.reshape(opt.episodes_per_batch,train_n_support, -1)


                emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

                logit_query,original_logits,completed_logits = FuseCosineNetHead(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot,x_gen,query_gen,normalize_=False)
                
                logit_query = scale * logit_query
                original_logits = scale * original_logits
                completed_logits = scale * completed_logits

                smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
                smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

                log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
                loss = -(smoothed_one_hot * log_prb).sum(dim=1)
                loss = loss.mean()

                acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))
                ori_acc = count_accuracy(original_logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
                com_acc = count_accuracy(completed_logits.reshape(-1, opt.test_way), labels_query.reshape(-1))

                train_accuracies.append(acc.item())
                original_acc.append(ori_acc.item())
                completed_acc.append(com_acc.item())
                train_losses.append(loss.item())

                if (i % 160 == 0):
                    train_acc_avg = np.mean(np.array(train_accuracies))
                    ori_acc_avg = np.mean(np.array(original_acc))
                    com_acc_avg = np.mean(np.array(completed_acc))
                    avg_loss = np.mean(np.array(train_losses))
                    print('Train Epoch: {}\tBatch: [{}]\tLoss: {:.2f}\tAccuracy: {:.2f} % ({:.2f} %)\tori_acc: {:.2f}\t com_acc: {:.2f}'.format(
                        epoch, i, avg_loss, train_acc_avg, acc,ori_acc_avg,com_acc_avg))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies