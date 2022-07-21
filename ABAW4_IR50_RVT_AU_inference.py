from __future__ import print_function

import os,sys,time,argparse,torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from models import IR_RVT_AU_plus_patch
from data import ABAW4_test
from utils.train import infer_FER_AUViT
from utils import parameter
from utils.utils import Logger
import torchvision.datasets as datasets

def parse_arguments():
    parser = argparse.ArgumentParser('argument for inference')

    parser.add_argument('--CUDA', type=str, default="0,1,2,3")
    
    parser.add_argument('--print_freq', type=int, default=10)

    parser.add_argument('--bs', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4)

    # model dataset
    parser.add_argument('--model', type=str, default='IR_RVT_AU')
    parser.add_argument('--dataset', type=str, default='ABAW4_test')
    parser.add_argument('--log', type=bool,default=False)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=args.CUDA

    args.model_path = './save/inference_{}'.format(args.dataset)
    args.result_file_name = ""

    args.save_folder = os.path.join(args.model_path)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    return args

def set_loader(args):
    mean = (0.5,0.5,0.5)
    std = (0.5, 0.5,0.5)
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize(size=(112, 112)),
        transforms.ToTensor(),
        normalize,
    ])

    test_folder=parameter.ABAW4_test_img_folder
    test_txt=parameter.ABAW4_test_txt
    test_dataset=ABAW4_test.load_ABAW4_test(test_txt, test_folder, val_transform)
    
    test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=args.bs, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    print('test dataset size:',test_dataset.__len__(), 'val_loader set size:', test_loader.__len__())
    return test_loader

def set_model(args):
    pthpath="save/ABAW4+Openface_models/ABAW4+Openface_IR50_RVT_AU_0.001_512_cosin_0/iter_99_0.68628.pth"
    model =IR_RVT_AU_plus_patch.IR50_ViT(num_classes=6,num_AU_patch=2)

    model.load_state_dict(torch.load(pthpath)["model"],strict=True)
    import datetime
    datetime_ = datetime.datetime.now()
    time_str=str(datetime_.date())+'-'+str(datetime_.time()).split('.')[0]
    args.result_file_name=pthpath.split("/")[-1][:-4]+'_'+time_str+".txt"
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    return model

def main():
    args = parse_arguments()
    if args.log:
        sys.stdout = Logger()
    print(args)
    # build data loader
    val_loader = set_loader(args)
    # build model
    model = set_model(args)
    # eval
    time1 = time.time()
    infer_FER_AUViT(val_loader, model, args)
    time2 = time.time()
    print('total time {:.2f}'.format(time2 - time1))

if __name__ == '__main__':
    main()
