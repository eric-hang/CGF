from get_model_data import get_model,get_dataset
from utils import set_gpu, Timer, count_accuracy, check_dir, log, loss_fn
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
def meta_test(opt, dataset_train, dataset_val, dataset_test, data_loader):
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

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "meta_test_log.txt")
    log(log_file_path, str(vars(opt)))
    (embedding_net, netE,netG ,propa_head, pre_head, cls_head) = get_model(opt)


    # Load saved model checkpoints
    saved_models = torch.load(os.path.join(opt.save_path, 'best_model_meta_val_{}w_{}s_{}.pth'.format(opt.test_way, opt.val_shot, opt.head)))
    netE.load_state_dict(saved_models['netE'])
    netE.eval()
    netG.load_state_dict(saved_models['netG'])
    netG.eval()

    embedding_net.load_state_dict(saved_models['embedding'])
    embedding_net.eval()
    propa_head.load_state_dict(saved_models['propa_head'])
    propa_head.eval()

    x_entropy = torch.nn.CrossEntropyLoss()

    # Evaluate on the validation split
    _, _, _, _, _= [x.eval() for x in (embedding_net, propa_head, cls_head, netE,netG)]
    test_accuracies = []
    original_acc = []
    completed_acc = []
    test_losses = []

    for i, batch in enumerate(tqdm(dloader_test(opt.seed)), 1):
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
            z = z.reshape(1,test_n_support, -1)
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            logit_query,original_logits,completed_logits = cls_head(k_all, propa_head, emb_query, emb_support, labels_support, opt.test_way, opt.val_shot, is_scale=True,sample_specific_feature=z,options=opt)

        loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
        acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
        ori_acc = count_accuracy(original_logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
        com_acc = count_accuracy(completed_logits.reshape(-1, opt.test_way), labels_query.reshape(-1))
        test_accuracies.append(acc.item())
        original_acc.append(ori_acc.item())
        completed_acc.append(com_acc.item())
        test_losses.append(loss.item())

    test_acc_avg = np.mean(np.array(test_accuracies))
    completed_acc_avg = np.mean(np.array(completed_acc))
    original_acc_avg = np.mean(np.array(original_acc))
    test_acc_ci95 = 1.96 * np.std(np.array(test_accuracies)) / np.sqrt(opt.val_episode)

    test_loss_avg = np.mean(np.array(test_losses))

    log(log_file_path, 'Test Loss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)\t orignial_acc: {:.2f}\t completed_acc: {:.2f}' \
        .format(test_loss_avg, test_acc_avg, test_acc_ci95,original_acc_avg,completed_acc_avg))
