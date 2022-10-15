from models.resnet12_2 import resnet12
import models.resnet12_pretrain as resnet12_pretrain
from models.meta_part_inference_mini import ProtoComNet
from models.PredTrainHead import LinearClassifier, LinearRotateHead
from models.model import Encoder,Generator
import models.resnet_inv as resnet_inv
def get_model(options):
    if options.dataset == 'miniImageNet' or options.dataset == 'CIFAR-FS':
        if options.use_trainval == 'True':
            n_classes=80
        else:
            n_classes=64
    elif options.dataset == 'tieredImageNet':
        n_classes = 351

    if options.network == "ResNet12":
        network = resnet12()
        fea_dim = options.feature_size
    elif options.network == 'ResNet12_pretrain':
        if options.dataset == 'miniImageNet':
            network = resnet12_pretrain.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_classes,opt=options)
        elif options.dataset == 'tieredImageNet':
            network = resnet12_pretrain.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_classes,opt=options)
        elif options.dataset == 'CIFAR-FS':
            network = resnet12_pretrain.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_classes,opt=options)
        fea_dim = options.feature_size
    elif options.network == 'ResNet12_inv':
        if options.dataset == 'tieredImageNet':
            network = resnet_inv.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_classes, no_trans=16, embd_size=64)
        elif options.dataset == 'miniImageNet':
            network = resnet_inv.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_classes, no_trans=16, embd_size=64)
        elif options.dataset == 'CIFAR-FS':
            network = resnet_inv.resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=n_classes, no_trans=16, embd_size=64)
        fea_dim = options.feature_size
    else:
        print ("Cannot recognize the network type")
        assert(False)
    network = network.cuda()
    netE = Encoder(opt = options).cuda()
    netG = Generator(opt = options).cuda()

    propa_head = ProtoComNet(opt=options, in_dim=fea_dim).cuda()

    if options.pre_head == 'LinearNet':
        pre_head = LinearClassifier(in_dim=fea_dim, n_classes=n_classes).cuda()
    elif options.pre_head == 'LinearRotateNet':
        pre_head = LinearRotateHead(in_dim=fea_dim, n_classes=n_classes).cuda()
    else:
        print("Cannot recognize the dataset type")
        assert (False)

    if options.phase == 'pretrain':
        from models.classification_heads_orgin import ClassificationHead
    else:
        from models.classification_heads import ClassificationHead
    # Choose the classification head
    if options.head == 'CosineNet':
        cls_head = ClassificationHead(base_learner='CosineNet').cuda()
    elif options.head == 'FuseCosNet':
        cls_head = ClassificationHead(base_learner='FuseCos').cuda()
    elif options.head == 'cls':
        cls_head = ClassificationHead(base_learner='FuseCos').cuda()
    else:
        cls_head = None
        print ("Cannot recognize the dataset type")
    return (network, netE,netG,propa_head, pre_head, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader, MiniImageNetPC
        if options.phase == 'data_scaler' or options.phase == 'savepart':
            dataset_train = MiniImageNet(phase='train', do_not_use_random_transf=True)
        elif options.phase == 'metainfer' or options.phase == 'pretrain_encoder':
            dataset_train = MiniImageNetPC(phase='train', shot=options.train_shot)
        elif options.phase == 'metatest' or options.phase == 'generative_test':
            dataset_train = None
        else:
            dataset_train = MiniImageNet(phase='train')


        if options.phase == 'pretrain_encoder':
            dataset_val = MiniImageNet(phase='val')
        elif options.phase == 'metatest' or options.phase == 'generative_test':
            dataset_val = None
        elif options.phase == 'data_scaler' or options.phase == 'savepart':
            dataset_val = MiniImageNet(phase='val', do_not_use_random_transf=True)
        else:
            dataset_val = MiniImageNet(phase='val')


        if options.phase == 'metatest' or options.phase == 'generative_test' :
            dataset_test = MiniImageNet(phase='test')
        elif options.phase == 'savepart':
            dataset_test = MiniImageNet(phase='test',do_not_use_random_transf=True)
        else:
            dataset_test = None
        data_loader = FewShotDataloader
    
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet,FewShotDataloader,tieredImageNetPC
        if options.phase == 'data_scaler' or options.phase == 'savepart':
            dataset_train = tieredImageNet(phase='train', do_not_use_random_transf=True)
        elif options.phase == 'metainfer' or options.phase == 'pretrain_encoder':
            dataset_train = tieredImageNetPC(phase='train', shot=options.train_shot)
        elif options.phase == 'metatest' or options.phase == 'generative_test':
            dataset_train = None
        else:
            dataset_train = tieredImageNet(phase='train')


        if options.phase == 'pretrain_encoder':
            dataset_val = tieredImageNet(phase='test')
        elif options.phase == 'metatest' or options.phase == 'generative_test':
            dataset_val = None
        elif options.phase == 'data_scaler' or options.phase == 'savepart':
            dataset_val = tieredImageNet(phase='val', do_not_use_random_transf=True)
        else:
            dataset_val = tieredImageNet(phase='test')


        if options.phase == 'metatest' or options.phase == 'generative_test':
            dataset_test = tieredImageNet(phase='test')
        else:
            dataset_test = None
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR-FS':
        from data.cifar_fs import CIFAR_FS,FewShotDataloader,CIFAR_FSPC
        if options.phase == 'data_scaler' or options.phase == 'savepart':
            dataset_train = CIFAR_FS(phase='train', do_not_use_random_transf=True)
        elif options.phase == 'metainfer' or options.phase == 'pretrain_encoder':
            dataset_train = CIFAR_FSPC(phase='train', shot=options.train_shot)
        elif options.phase == 'metatest' or options.phase == 'generative_test':
            dataset_train = None
        else:
            dataset_train = CIFAR_FS(phase='train')


        if options.phase == 'pretrain_encoder':
            dataset_val = CIFAR_FS(phase='val')
        elif options.phase == 'metatest' or options.phase == 'generative_test':
            dataset_val = None
        elif options.phase == 'data_scaler' or options.phase == 'savepart':
            dataset_val = CIFAR_FS(phase='val', do_not_use_random_transf=True)
        else:
            dataset_val = CIFAR_FS(phase='val')


        if options.phase == 'metatest' or options.phase == 'generative_test' or options.phase == 'savepart':
            dataset_test = CIFAR_FS(phase='test')
        else:
            dataset_test = None
        data_loader = FewShotDataloader

    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    return (dataset_train, dataset_val, dataset_test, data_loader)