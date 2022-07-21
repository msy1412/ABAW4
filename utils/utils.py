import torch,bcolz,io,os,sys
import torchvision.transforms as transforms
import torch.nn.functional as F

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image

from torch.optim.lr_scheduler import _LRScheduler

# Support: ['get_time', 'l2_norm', 'make_weights_for_balanced_classes', 'get_val_pair', 'get_val_data', 'separate_irse_bn_paras', 'separate_resnet_bn_paras', 'warm_up_lr', 'schedule_lr', 'de_preprocess', 'hflip_batch', 'ccrop_batch', 'gen_plot', 'perform_val', 'buffer_val', 'AverageMeter', 'accuracy']


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cfp_ff, cfp_ff_issame = get_val_pair(data_path, 'cfp_ff')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
    vgg2_fp, vgg2_fp_issame = get_val_pair(data_path, 'vgg2_fp')

    return lfw, cfp_ff, cfp_fp, agedb_30, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_30_issame, calfw_issame, cplfw_issame, vgg2_fp_issame


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

    # print(optimizer)


def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)


def de_preprocess(tensor):

    return tensor * 0.5 + 0.5


hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


ccrop = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([128, 128]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def ccrop_batch(imgs_tensor):
    ccropped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        ccropped_imgs[i] = ccrop(img_ten)

    return ccropped_imgs


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf


def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, carray, issame, nrof_folds = 10, tta = True):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                embeddings[idx:idx + batch_size] = l2_norm(backbone(ccropped.to(device))).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                ccropped = ccrop_batch(batch)
                fliped = hflip_batch(ccropped)
                emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                ccropped = ccrop_batch(batch)
                embeddings[idx:] = l2_norm(backbone(ccropped.to(device))).cpu()

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch):
    writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
    writer.add_scalar('{}_Best_Threshold'.format(db_name), best_threshold, epoch)
    writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred    = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))

#     return res


class Logger(object):
    def __init__(self, filename="log.txt"):
        import time
        localtime = time.localtime(time.time())
        f = ""
        for i in range(0,5):
            f += str(localtime[i]) + "_"
        filename="log/"+f+filename
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
       
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def save_model(model, optimizer, opt, epoch, save_file,without_module=True):
    print('==> Saving...')
    if without_module:
        state = {
            'opt': opt,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    else:
        state = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    torch.save(state, save_file)
    del state

def set_optimizer(opt, model):

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=opt.lr,
                              momentum=opt.momentum,
                              weight_decay=opt.wd)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                               lr=opt.lr,
                               weight_decay=opt.wd)
    else:
        raise ValueError('optimizer not supported: {}'.format(opt.optimizer))

    return optimizer

def load_weights_dropModule(model,model_path,state_dict_keyword='model_state_dict'):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint[state_dict_keyword]
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def load_weights_Spvs_AU_dropModule(model,model_path,Spvs_after_layer,state_dict_keyword='model_state_dict'):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint[state_dict_keyword]
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_name = k[7:] # remove `module.`
        if new_name.startswith( 'block' ):
            name=new_name.split('.')
            layer_number=int(name[1])
            if layer_number<=(Spvs_after_layer-1):
                name[0]="blocks_1"
            elif layer_number>(Spvs_after_layer-1):
                name[0]="blocks_2"
                name[1]=str(layer_number-Spvs_after_layer)
            new_name='.'.join(str(d) for d in name)
        
        new_state_dict[new_name] = v

    # load params
    #!!!remember to use strict=True to check which weight has been loaded and which not!!!
    model.load_state_dict(new_state_dict,strict=False)
    return model
