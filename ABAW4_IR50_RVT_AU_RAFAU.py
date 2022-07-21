from __future__ import print_function

import os,sys,time,argparse,timm,torch
from random import triangular
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils import data
from models import IR,IR_RVT_AU_plus_patch
from utils.train import train_iter_with_AU
from data.ABAW4_FER_AU import ABAW_FER_AU
from data import RAFAU_100
from utils import parameter
from utils.utils import set_optimizer,load_weights_dropModule,WarmUpLR,Logger
import torchvision.datasets as datasets
from utils import sample

def parse_arguments():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--CUDA', type=str, default="4,5,6,7")
    
    parser.add_argument('--valid_freq', type=int, default=100)

    parser.add_argument('--AU_cls', type=int, default=21)

    parser.add_argument('--print_num', type=int, default=460)

    parser.add_argument('--bs', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--warmup', type=float, default=0)
    # optimization
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['adam','sgd'])
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--scheduler', type=str ,default="cosin",
                        choices=['cosin','step','none'])
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay')#5e-4,
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # model dataset
    parser.add_argument('--model', type=str, default='IR50_RVT_AU')
    parser.add_argument('--dataset', type=str, default='ABAW4+RAFAU')
    parser.add_argument('--save_model', type=bool,default=True)
    parser.add_argument('--log', type=bool,default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.CUDA

    args.model_path = './save/{}_models'.format(args.dataset)
    args.model_name = '{}_{}_{}_{}_{}_{}'.\
        format(args.dataset, args.model, args.lr, args.bs,args.scheduler,args.warmup)
    print(f'model name: {args.model_name}')
    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args

def set_loader(args):
    mean = (0.5,0.5,0.5)
    std = (0.5, 0.5,0.5)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize(size=(112, 112)),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(112, 112)),
        transforms.ToTensor(),
        normalize,
    ])

    train_folder=parameter.ABAW4_train_img_folder
    valid_folder=parameter.ABAW4_valid_img_folder
    train_FER_AU_CSV=parameter.ABAW4_FER_AU_csv
    
    train_dataset=datasets.ImageFolder(train_folder, train_transform)
    # train_dataset=ABAW_FER_AU(train_FER_AU_CSV,train_folder,train_transform)
    val_dataset=datasets.ImageFolder(valid_folder, val_transform)

    print('ABAW4 train set size:', train_dataset.__len__())
    print('ABAW4 validation set size:', val_dataset.__len__())
    # train_sampler = weighted_sampler_generator(data_txt_dir, args.dataset)
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.bs, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.bs, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    print('train_loader set size:', train_loader.__len__())
    print('valid_loader set size:', val_loader.__len__())

    RAFAU_img_folder_path=parameter.AU_img_folder_path
    RAFAU_train_list_file = parameter.AU_train_list_file
    RAFAU_test_list_file = parameter.AU_test_list_file
    
    RAFAU_train_dataset =  RAFAU_100.load_RAFAU(RAFAU_train_list_file,RAFAU_img_folder_path,transform=train_transform,phase="train")
    RAFAU_train_loader = data.DataLoader(RAFAU_train_dataset,batch_size=int(args.bs/4), shuffle=True,num_workers=args.num_workers, pin_memory=True)
    RAFAU_test_dataset =  RAFAU_100.load_RAFAU(RAFAU_test_list_file,RAFAU_img_folder_path,transform=val_transform,phase="train")
    RAFAU_test_loader = data.DataLoader(RAFAU_test_dataset,batch_size=int(args.bs/4), shuffle=True,num_workers=args.num_workers, pin_memory=True)
    print('RAFAU_Train set size:', RAFAU_train_dataset.__len__())
    print('RAFAU_Test set size:', RAFAU_test_dataset.__len__())

    return train_loader,RAFAU_train_loader,val_loader


def set_model(args):
    pthpath="/home/sztu/msy_Project/save_model/backbone_ir50_ms1m_epoch63.pth"
    model =IR_RVT_AU_plus_patch.IR50_ViT(num_classes=6,ir_50_pth=pthpath,num_AU_patch=2)

    criterion_EXPR = torch.nn.CrossEntropyLoss()
    criterion_AU = torch.nn.BCEWithLogitsLoss()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion_EXPR = criterion_EXPR.cuda()
        criterion_AU = criterion_AU.cuda()
        cudnn.benchmark = True

    return model,[criterion_EXPR, criterion_AU]

def main():
    args = parse_arguments()
    if args.log:
        sys.stdout = Logger()
    print(args)
 
    # build data loader
    train_loader,AU_loader,val_loader = set_loader(args)

    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, model)
    warmup_scheduler=None
    all_iteration=args.epochs*train_loader.__len__()
    warmup_iter=0 
    if args.warmup>0:
        warmup_iter = args.warmup*train_loader.__len__()
        warmup_scheduler = WarmUpLR(optimizer,warmup_iter)

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
        print("scheduler : multistep")
    elif args.scheduler == 'cosin':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=all_iteration-warmup_iter,eta_min=1e-6)
        print("scheduler : cosin")

    # train for iteration
    train_iter_with_AU(train_loader,AU_loader,val_loader,None,model,criterion, optimizer,
                    warmup_scheduler,all_iteration, warmup_iter,scheduler, args,AU_RATIO=1)
if __name__ == '__main__':
    main()
