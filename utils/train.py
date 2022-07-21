import time,sys,os,torch
import numpy as np
from utils import AverageMeter, accuracy
from utils.evaluation_metric import AU_metric, EXPR_metric
from utils.utils import save_model

def train_with_AU(FER_train_loader,AU_train_loader, model, criterion, optimizer, epoch, warmup_epoch, AU_RATIO,args):
    """one epoch training"""
    print_freq=int(FER_train_loader.__len__()/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}
    iter_AU_loader=iter(AU_train_loader)
    end = time.time()

    for idx, (images, FER_targets) in enumerate(FER_train_loader):
        data_time.update(time.time() - end)

        # load AU data
        try:
          AU_img,AU_targets=next(iter_AU_loader)
        except:
          iter_AU_loader=iter(AU_train_loader)
          AU_img,AU_targets=next(iter_AU_loader)
        # concat FER and AU imgs
        total_img=torch.cat((images,AU_img),dim=0)
        # process FER labels
        FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
        FER_targets= FER_targets.cuda()
        # process AU labels
        AU_target_arr = np.array(AU_targets,dtype='int32').T
        AU_target_tensor = torch.tensor(AU_target_arr)
        AU_targets = AU_target_tensor.cuda()
        #batch-size
        FER_bsz = len(FER_targets)
        AU_bsz = len(AU_targets)
        whole_bsz=FER_bsz+AU_bsz

        # imgs to cuda
        total_img = total_img.cuda()
        # model
        output = model(total_img)


        FER_output = output[0][0:FER_bsz]
        AU_output = output[1][FER_bsz:]
        #AU_output = nn.Parameter(torch.ones(AU_bsz, 21)).cuda()
        loss_EXPR = criterion[0](FER_output, FER_targets)
        loss_AU = criterion[1](AU_output, AU_targets.float())
        loss = loss_EXPR + AU_RATIO * loss_AU


        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        losses.update(loss, whole_bsz)
        acc_EXPR_batch = accuracy(FER_output, FER_targets)
        acc_EXPR.update(acc_EXPR_batch[0], FER_bsz)
        label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
        label_EXPR['pred'].append(FER_output.cpu().detach().numpy())
        predict_AU = torch.sigmoid(AU_output)
        predict_AU = torch.round(predict_AU)
        correct_sum = sum(predict_AU == AU_targets).sum()
        acc_AU_batch = correct_sum.float()/(AU_bsz*args.AU_cls)
        acc_AU.update(acc_AU_batch, AU_bsz)
        label_AU['gt'].append(AU_targets.cpu().detach().numpy())
        label_AU['pred'].append(predict_AU.cpu().detach().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch <= warmup_epoch and args.warmup and (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                  'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'.format(
                epoch, idx + 1, len(FER_train_loader),
                # batch_time=batch_time,data_time=data_time, 
                loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
            sys.stdout.flush()
        elif (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                  'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'
                  'lr {lr:.8f}\t'.format(
                epoch, idx + 1, len(FER_train_loader),
                # batch_time=batch_time,data_time=data_time, 
                loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR, acc_AU=acc_AU))
            sys.stdout.flush()

    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]
    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = AU_metric(label_pred, label_gt)
    AU_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs, AU_accs

def train_FER(FER_train_loader, model, criterion, optimizer,epoch,warmup_epoch, args):
    """one epoch training"""
    print_freq=int(FER_train_loader.__len__()/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}


    end = time.time()

    for idx, (images, targets) in enumerate(FER_train_loader):
        data_time.update(time.time() - end)

        # #warm up
        # if epoch <= warmup_epoch and args.warmup:
        #     warmup_scheduler.step()
        #     warm_lr = warmup_scheduler.get_lr()
        #     # print("warm_lr:%s" % warm_lr)

        targets = targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        targets = torch.as_tensor(targets, dtype=torch.int64)
        images = images.cuda()
        targets= targets.cuda()

        bsz = targets.shape[0]

        # model
        output = model(images)
        loss_EXPR = criterion(output, targets)

        # optimize
        optimizer.zero_grad()
        loss_EXPR.backward()
        optimizer.step()

        # update metric
        losses.update(loss_EXPR, bsz)
        acc_EXPR_batch = accuracy(output, targets)
        acc_EXPR.update(acc_EXPR_batch[0], bsz)
        label_EXPR['gt'].append(targets.cpu().detach().numpy())
        label_EXPR['pred'].append(output.cpu().detach().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if  epoch <= warmup_epoch and args.warmup and (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'.format(
                epoch, idx + 1, len(FER_train_loader),loss=losses, acc_EXPR=acc_EXPR))
                # 'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # 'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #  batch_time=batch_time,data_time=data_time,
            sys.stdout.flush()
        elif (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                'lr {lr:.8f}\t'.format(
                epoch, idx + 1, len(FER_train_loader), loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR))
            sys.stdout.flush() 


    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs

def train_FERAU(FER_train_loader,model, criterion, optimizer,epoch,warmup_epoch, args,AU_RATIO=1):
    """one epoch training"""
    print_freq=int(FER_train_loader.__len__()/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}

    end = time.time()

    for idx, (images, targets) in enumerate(FER_train_loader):
        data_time.update(time.time() - end)

        FER_targets,AU_targets = targets[0],targets[1]
        # process FER labels
        FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
        FER_targets= FER_targets.cuda()
        # process AU labels
        AU_target_arr = np.array(AU_targets,dtype='int32')
        AU_target_tensor = torch.tensor(AU_target_arr)
        AU_targets = AU_target_tensor.cuda()
        #batch-size
        bsz = len(FER_targets)


        # imgs to cuda
        images = images.cuda()
        # model
        output = model(images)

        FER_output = output[0]
        AU_output = output[1]
        #AU_output = nn.Parameter(torch.ones(AU_bsz, 21)).cuda()
        loss_EXPR = criterion[0](FER_output, FER_targets)
        loss_AU = criterion[1](AU_output, AU_targets.float())
        loss = loss_EXPR + AU_RATIO * loss_AU


        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        losses.update(loss, bsz)
        acc_EXPR_batch = accuracy(FER_output, FER_targets)
        acc_EXPR.update(acc_EXPR_batch[0], bsz)
        label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
        label_EXPR['pred'].append(FER_output.cpu().detach().numpy())
        predict_AU = torch.sigmoid(AU_output)
        predict_AU = torch.round(predict_AU)
        correct_sum = sum(predict_AU == AU_targets).sum()
        acc_AU_batch = correct_sum.float()/(bsz*args.AU_cls)
        acc_AU.update(acc_AU_batch, bsz)
        label_AU['gt'].append(AU_targets.cpu().detach().numpy())
        label_AU['pred'].append(predict_AU.cpu().detach().numpy())
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch <= warmup_epoch and args.warmup and (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                  'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'.format(
                epoch, idx + 1, len(FER_train_loader),
                # batch_time=batch_time,data_time=data_time, 
                loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
            sys.stdout.flush()
        elif (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                  'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'
                  'lr {lr:.8f}\t'.format(
                epoch, idx + 1, len(FER_train_loader),
                # batch_time=batch_time,data_time=data_time, 
                loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR, acc_AU=acc_AU))
            sys.stdout.flush()

    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]
    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = AU_metric(label_pred, label_gt)
    AU_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs, AU_accs

def train_FER_with_FER_iter(FER_train_loader,aux_train_loader, model, criterion, optimizer,epoch,warmup_epoch, args):
    """one epoch training"""
    print_freq=int(FER_train_loader.__len__()/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    iter_aux_loader=iter(aux_train_loader)

    end = time.time()

    for idx, (images, targets) in enumerate(FER_train_loader):
        data_time.update(time.time() - end)

        # #warm up
        # if epoch <= warmup_epoch and args.warmup:
        #     warmup_scheduler.step()
        #     warm_lr = warmup_scheduler.get_lr()
        #     # print("warm_lr:%s" % warm_lr)

        # load aux data
        try:
          aux_img,aux_targets=next(iter_aux_loader)
        except:
          iter_aux_loader=iter(aux_train_loader)
          aux_img,aux_targets=next(iter_aux_loader)

        total_img=torch.cat((images,aux_img),dim=0)
        # process main labels
        targets = targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        targets = torch.as_tensor(targets, dtype=torch.int64)
        targets= targets.cuda()
        # process aux labels
        aux_targets = aux_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        aux_targets = torch.as_tensor(aux_targets, dtype=torch.int64)
        aux_targets= aux_targets.cuda()

        # concat main and aux targets 
        total_target=torch.cat((targets,aux_targets),dim=0)
        #batch-size
        main_bsz = len(targets)
        aux_bsz = len(aux_targets)
        whole_bsz=main_bsz+aux_bsz
        # imgs to cuda
        total_img = total_img.cuda()
        # model
        output = model(total_img)
        loss_EXPR = criterion(output, total_target)

        # optimize
        optimizer.zero_grad()
        loss_EXPR.backward()
        optimizer.step()

        # update metric
        losses.update(loss_EXPR, whole_bsz)
        acc_EXPR_batch = accuracy(output, total_target)
        acc_EXPR.update(acc_EXPR_batch[0], whole_bsz)
        label_EXPR['gt'].append(total_target.cpu().detach().numpy())
        label_EXPR['pred'].append(output.cpu().detach().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if  epoch <= warmup_epoch and args.warmup and (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'.format(
                epoch, idx + 1, len(FER_train_loader),loss=losses, acc_EXPR=acc_EXPR))
                # 'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # 'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #  batch_time=batch_time,data_time=data_time,
            sys.stdout.flush()
        elif (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                'lr {lr:.8f}\t'.format(
                epoch, idx + 1, len(FER_train_loader), loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR))
            sys.stdout.flush() 


    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs

def validate_with_AU(AU_val_loader,FER_val_loader, model, criterion,AU_RATIO, args):
    """validation"""
    model.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}
    iter_AU_loader=iter(AU_val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, (images, FER_targets) in enumerate(FER_val_loader):
            data_time.update(time.time() - end)
            # load AU data
            try:
                AU_img,AU_targets=next(iter_AU_loader)
            except:
                iter_AU_loader=iter(AU_val_loader)
                AU_img,AU_targets=next(iter_AU_loader)
            # concat FER and AU imgs
            total_img=torch.cat((images,AU_img),dim=0)
            # process FER labels
            FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
            FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
            FER_targets= FER_targets.cuda()
            # process AU labels
            AU_target_arr = np.array(AU_targets,dtype='int32').T
            AU_target_tensor = torch.tensor(AU_target_arr)
            AU_targets = AU_target_tensor.cuda()
            #batch-size
            FER_bsz = len(FER_targets)
            AU_bsz = len(AU_targets)
            whole_bsz=FER_bsz+AU_bsz
    
            # imgs to cuda
            total_img = total_img.cuda()
            # model
            output = model(total_img)
            FER_output = output[0][0:FER_bsz]
            AU_output = output[1][FER_bsz:]
            #AU_output = nn.Parameter(torch.ones(AU_bsz, 21)).cuda()
            loss_EXPR = criterion[0](FER_output, FER_targets)
            loss_AU = criterion[1](AU_output, AU_targets.float())
            loss =   loss_EXPR + AU_RATIO * loss_AU
    
    
            # update metric
            losses.update(loss, whole_bsz)
            acc_EXPR_batch = accuracy(FER_output, FER_targets)
            acc_EXPR.update(acc_EXPR_batch[0], FER_bsz)
            label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
            label_EXPR['pred'].append(FER_output.cpu().detach().numpy())
            predict_AU = torch.sigmoid(AU_output)
            predict_AU = torch.round(predict_AU)
            correct_sum = sum(predict_AU == AU_targets).sum()
            acc_AU_batch = correct_sum.float()/(AU_bsz*args.AU_cls)
            acc_AU.update(acc_AU_batch, AU_bsz)
            label_AU['gt'].append(AU_targets.cpu().detach().numpy())
            label_AU['pred'].append(predict_AU.cpu().detach().numpy())
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            # if (idx + 1) % args.print_freq == 0:
            #     print('test: [{0}/{1}]\t'
            #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
            #           'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
            #           'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'.format(
            #         idx + 1, len(FER_val_loader), batch_time=batch_time,
            #         loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
            #     sys.stdout.flush()

    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]
    label_gt = np.concatenate(label_AU['gt'], axis=0)
    label_pred = np.concatenate(label_AU['pred'], axis=0)
    f1, acc, total_acc = AU_metric(label_pred, label_gt)
    AU_accs = [f1, acc, total_acc]


    return losses.avg, EXPR_accs, AU_accs

def validate_FER(val_loader, model, criterion, args):
    """validation"""
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    with torch.no_grad():
        end = time.time()
        for idx, (images, targets) in enumerate(val_loader):
            targets = targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
            targets = torch.as_tensor(targets, dtype=torch.int64)
            images = images.cuda()
            targets= targets.cuda()
  
            # bsz = len(targets)  
            bsz = targets.shape[0]

            # model
            output = model(images)
            if len(output)==2:
                output=output[0]
            loss_EXPR = criterion(output, targets)


            # update metric
            losses.update(loss_EXPR, bsz)
            acc_EXPR_batch = accuracy(output, targets)
            acc_EXPR.update(acc_EXPR_batch[0], bsz)
            label_EXPR['gt'].append(targets.cpu().detach().numpy())
            label_EXPR['pred'].append(output.cpu().detach().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if idx % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'.format(
            #            idx, len(val_loader), batch_time=batch_time,
            #            loss=losses, acc_EXPR=acc_EXPR))
            #     sys.stdout.flush()

    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs

def validate_2_FER(A_val_loader,B_val_loader, model, criterion, args):
    """validation"""
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i,(data_set) in enumerate([A_val_loader,B_val_loader]):
            batch_time = AverageMeter()
            losses = AverageMeter()
            acc_EXPR = AverageMeter()
            label_EXPR = {'gt': [], 'pred': []}
            for idx, (images, targets) in enumerate(data_set):
                targets = targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
                targets = torch.as_tensor(targets, dtype=torch.int64)
                images = images.cuda()
                targets= targets.cuda()
    
                # bsz = len(targets)  
                bsz = targets.shape[0]

                # model
                output = model(images)
                loss_EXPR = criterion(output, targets)

                # update metric
                # losses.update(loss_EXPR, bsz)
                acc_EXPR_batch = accuracy(output, targets)
                acc_EXPR.update(acc_EXPR_batch[0], bsz)
                label_EXPR['gt'].append(targets.cpu().detach().numpy())
                label_EXPR['pred'].append(output.cpu().detach().numpy())


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if idx % args.print_freq == 0:
                #     print('Test: [{0}/{1}]\t'
                #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #         'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'.format(
                #             idx, len(data_set), batch_time=batch_time,
                #             acc_EXPR=acc_EXPR))
                #     sys.stdout.flush()
            if i==0:
                label_gt = np.concatenate(label_EXPR['gt'], axis=0)
                label_pred = np.concatenate(label_EXPR['pred'], axis=0)
                f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
                A_accs = [f1, acc, total_acc]
            else:
                label_gt = np.concatenate(label_EXPR['gt'], axis=0)
                label_pred = np.concatenate(label_EXPR['pred'], axis=0)
                f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
                B_accs = [f1, acc, total_acc]

    return A_accs,B_accs

def train_iter_FER(FER_train_loader,FER_val_loader,model, criterion, optimizer,warmup_scheduler, all_iterations, warmup_iter,scheduler, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    end = time.time()
    iter_FER_loader=iter(FER_train_loader)

    for iteration in range(all_iterations):

        data_time.update(time.time() - end)

        # load data
        try:
          images, targets=next(iter_FER_loader)
        except:
          iter_FER_loader=iter(FER_train_loader)
          images, targets=next(iter_FER_loader)

        # process FER labels
        targets = targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        targets = torch.as_tensor(targets, dtype=torch.int64)
        targets= targets.cuda()

        #batch-size
        bsz = len(targets)

        # imgs to cuda
        images = images.cuda()
        # model
        output = model(images)

        loss_EXPR = criterion(output, targets)

        # optimize
        optimizer.zero_grad()
        loss_EXPR.backward()
        optimizer.step()

        # update metric
        # update metric
        losses.update(loss_EXPR, bsz)
        acc_EXPR_batch = accuracy(output, targets)
        acc_EXPR.update(acc_EXPR_batch[0], bsz)
        label_EXPR['gt'].append(targets.cpu().detach().numpy())
        label_EXPR['pred'].append(output.cpu().detach().numpy())

        #warm up
        if args.warmup>0 and iteration <= warmup_iter :
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()
            # print("warm_lr:%s" % round(warm_lr[0],5))

        elif args.scheduler!='none':
            scheduler.step()  # update lr
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if(iteration + 1) % args.print_freq == 0:
            if iteration <= warmup_iter and args.warmup>0 :
                print('Train:[{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses, acc_EXPR=acc_EXPR))
                sys.stdout.flush()
                
            else:
                print('Train:[{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'lr {lr:.8f}\t'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR))

        if(iteration + 1) % args.valid_freq == 0:
            # eval
            time3 = time.time()
            loss, EXPR_accs= validate_FER(FER_val_loader, model, criterion, args)
            time4 = time.time()
            print('Validation iter {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
                iteration, time4 - time3, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2]))
            if args.save_model:
                if EXPR_accs[0]>= 0.75:
                    save_file = os.path.join(
                        args.save_folder, 'iter_{iter}_F1_{f1}.pth'.format(iter=iteration,f1=round(EXPR_accs[0],5)))
                    save_model(model, optimizer, args, iteration, save_file)
            sys.stdout.flush()

        model.train()

def train_iter_with_AU(FER_train_loader,AU_train_loader, FER_val_loader,AU_test_loader,model, criterion, optimizer,warmup_scheduler, all_iterations, warmup_iter,scheduler, args,AU_RATIO=1):
    """one epoch training"""
    print_freq=int(all_iterations/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}
    iter_FER_loader=iter(FER_train_loader)
    iter_AU_loader=iter(AU_train_loader)
    end = time.time()

    for iteration in range(all_iterations):
        data_time.update(time.time() - end)
        # load AU data
        try:
          FER_images, FER_targets=next(iter_FER_loader)
        except:
          iter_FER_loader=iter(FER_train_loader)
          FER_images, FER_targets=next(iter_FER_loader)
        try:
          AU_img,AU_targets=next(iter_AU_loader)
        except:
          iter_AU_loader=iter(AU_train_loader)
          AU_img,AU_targets=next(iter_AU_loader)
        # concat FER and AU imgs
        total_img=torch.cat((FER_images,AU_img),dim=0)
        # process FER labels
        FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
        FER_targets= FER_targets.cuda()
        # process AU labels
        AU_target_arr = np.array(AU_targets,dtype='int32').T
        AU_target_tensor = torch.tensor(AU_target_arr)
        AU_targets = AU_target_tensor.cuda()
        #batch-size
        FER_bsz = len(FER_targets)
        AU_bsz = len(AU_targets)
        whole_bsz=FER_bsz+AU_bsz

        # imgs to cuda
        total_img = total_img.cuda()
        # model
        output = model(total_img)

        FER_output = output[0][0:FER_bsz]
        AU_output = output[1][FER_bsz:]
        #AU_output = nn.Parameter(torch.ones(AU_bsz, 21)).cuda()
        loss_EXPR = criterion[0](FER_output, FER_targets)
        loss_AU = criterion[1](AU_output, AU_targets.float())
        loss = loss_EXPR + AU_RATIO * loss_AU


        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        losses.update(loss, whole_bsz)
        acc_EXPR_batch = accuracy(FER_output, FER_targets)
        acc_EXPR.update(acc_EXPR_batch[0], FER_bsz)
        label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
        label_EXPR['pred'].append(FER_output.cpu().detach().numpy())
        predict_AU = torch.sigmoid(AU_output)
        predict_AU = torch.round(predict_AU)
        correct_sum = sum(predict_AU == AU_targets).sum()
        acc_AU_batch = correct_sum.float()/(AU_bsz*args.AU_cls)
        acc_AU.update(acc_AU_batch, AU_bsz)
        label_AU['gt'].append(AU_targets.cpu().detach().numpy())
        label_AU['pred'].append(predict_AU.cpu().detach().numpy())
        
        #warm up
        if args.warmup>0 and iteration <= warmup_iter :
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()
            # print("warm_lr:%s" % round(warm_lr[0],5))

        elif args.scheduler :
            scheduler.step()  # update lr
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if(iteration + 1) % print_freq == 0:
            if iteration <= warmup_iter and args.warmup>0 :
                print('Train: [{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
                sys.stdout.flush()

            else:
                print('Train: [{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'
                    'lr {lr:.8f}\t'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR, acc_AU=acc_AU))

        if(iteration + 1) % args.valid_freq == 0:
            # eval
            time3 = time.time()
            if AU_test_loader!=None:
                loss, EXPR_accs, AU_accs= validate_with_AU(AU_test_loader,FER_val_loader, model, criterion,AU_RATIO, args)
                time4 = time.time()
                print('Validation iter {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}, AU F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
                    iteration, time4 - time3, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2], AU_accs[0], AU_accs[1], AU_accs[2]))
            else:
                loss,EXPR_accs= validate_FER(FER_val_loader,model, criterion[0], args)
                time4 = time.time()
                print('Validation iter {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
                    iteration+1, time4 - time3, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2]))
            if args.save_model:
                if EXPR_accs[0]>= 0.63:
                    save_file = os.path.join(
                        args.save_folder, 'iter_{iter}_F1_{acc}.pth'.format(iter=iteration,acc=round(EXPR_accs[0],5)))
                    save_model(model, optimizer, args, iteration, save_file)

            sys.stdout.flush()
        model.train()

def train_iter_2_FER(main_train_loader,aux_train_loader,main_test_loader,aux_test_loader, model, criterion, optimizer,warmup_scheduler, all_iterations, warmup_iter,scheduler, args):
    """one epoch training"""
    # print_freq=int(all_iterations/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_aux = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_aux = {'gt': [], 'pred': []}
    iter_FER_loader=iter(main_train_loader)
    iter_aux_loader=iter(aux_train_loader)
    end = time.time()

    for iteration in range(all_iterations):
        data_time.update(time.time() - end)
        # load main data
        try:
          FER_images, FER_targets=next(iter_FER_loader)
        except:
          iter_FER_loader=iter(main_train_loader)
          FER_images, FER_targets=next(iter_FER_loader)
        # load aux data
        try:
          aux_img,aux_targets=next(iter_aux_loader)
        except:
          iter_aux_loader=iter(aux_train_loader)
          aux_img,aux_targets=next(iter_aux_loader)
        # concat FER imgs 
        total_img=torch.cat((FER_images,aux_img),dim=0)
        
        # process FER labels
        FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
        FER_targets= FER_targets.cuda()
        # process aux labels
        aux_targets = aux_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        aux_targets = torch.as_tensor(aux_targets, dtype=torch.int64)
        aux_targets= aux_targets.cuda()
        # concat FER and AU targets 
        total_target=torch.cat((FER_targets,aux_targets),dim=0)
        #batch-size
        FER_bsz = len(FER_targets)
        aux_bsz = len(aux_targets)
        whole_bsz=FER_bsz+aux_bsz

        # imgs to cuda
        total_img = total_img.cuda()
        # model
        output = model(total_img)
        #loss
        loss = criterion(output, total_target)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        FER_output=output[:FER_bsz]
        aux_output=output[FER_bsz:]
        losses.update(loss, whole_bsz)
        acc_EXPR_batch = accuracy(FER_output, FER_targets)
        acc_EXPR.update(acc_EXPR_batch[0], FER_bsz)
        label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
        label_EXPR['pred'].append(FER_output.cpu().detach().numpy())

        acc_aux_batch = accuracy(aux_output, aux_targets)
        acc_aux.update(acc_aux_batch[0], aux_bsz)
        label_aux['gt'].append(aux_targets.cpu().detach().numpy())
        label_aux['pred'].append(aux_output.cpu().detach().numpy())
        
        #warm up
        if args.warmup>0 and iteration <= warmup_iter :
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()
            # print("warm_lr:%s" % round(warm_lr[0],5))
        # update learning rate
        elif args.scheduler != None :
            scheduler.step()  # update lr
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if(iteration + 1) % args.print_freq == 0:
            if iteration <= warmup_iter and args.warmup>0 :
                print('Train: [{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'Accaux {acc_aux.val:.3f} ({acc_aux.avg:.3f})\t'
                    'warm_lr {warm_lr:.7f}'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses, acc_EXPR=acc_EXPR, acc_aux=acc_aux,warm_lr=warm_lr[0]))
                sys.stdout.flush()

            else:
                print('Train: [{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'Accaux {acc_aux.val:.3f} ({acc_aux.avg:.3f})\t'
                    'lr {lr:.8f}\t'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR, acc_aux=acc_aux))

        if(iteration + 1) % args.valid_freq == 0:
            # eval
            time3 = time.time()
            if aux_test_loader!=None:
                EXPR_accs, aux_accs= validate_2_FER(main_test_loader,aux_test_loader, model, criterion, args)
                time4 = time.time()
                print('Validation iter {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}, aux F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
                    iteration+1, time4 - time3, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2], aux_accs[0], aux_accs[1], aux_accs[2]))
            else:
                loss,EXPR_accs= validate_FER(main_test_loader,model, criterion, args)
                time4 = time.time()
                print('Validation iter {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
                    iteration+1, time4 - time3, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2]))

            if args.save_model:
                if EXPR_accs[0]>= 0.73:
                    save_file = os.path.join(
                        args.save_folder, 'iter_{iter}_F1_{f1}.pth'.format(iter=iteration,f1=round(EXPR_accs[0],5)))
                    save_model(model, optimizer, args, iteration, save_file)

            sys.stdout.flush()
        model.train()

def train_iter_AffectNetFERAU(FER_train_loader,FER_val_loader,model, criterion, optimizer,warmup_scheduler, all_iterations, warmup_iter,scheduler, args,AU_RATIO=1):
    """one epoch training"""
    print_freq=int(all_iterations/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}
    iter_FER_loader=iter(FER_train_loader)
    end = time.time()

    for iteration in range(all_iterations):
        data_time.update(time.time() - end)
        # load AU data
        try:
          FER_images, FER_targets,AU_targets = next(iter_FER_loader)
        except:
          iter_FER_loader=iter(FER_train_loader)
          FER_images, FER_targets, AU_targets =next(iter_FER_loader)

        # process FER labels
        FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
        FER_targets= FER_targets.cuda()
        # process AU labels
        AU_target_arr = np.array(AU_targets,dtype='int32').T
        AU_target_tensor = torch.tensor(AU_target_arr)
        AU_targets = AU_target_tensor.cuda()
        #batch-size
        bsz = len(FER_targets)


        # imgs to cuda
        FER_images = FER_images.cuda()
        # model
        output = model(FER_images)

        FER_output = output[0]
        AU_output = output[1]
        #AU_output = nn.Parameter(torch.ones(AU_bsz, 21)).cuda()
        loss_EXPR = criterion[0](FER_output, FER_targets)
        loss_AU = criterion[1](AU_output, AU_targets.float())
        loss = loss_EXPR + AU_RATIO * loss_AU


        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        losses.update(loss, bsz)
        acc_EXPR_batch = accuracy(FER_output, FER_targets)
        acc_EXPR.update(acc_EXPR_batch[0], bsz)
        label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
        label_EXPR['pred'].append(FER_output.cpu().detach().numpy())
        predict_AU = torch.sigmoid(AU_output)
        predict_AU = torch.round(predict_AU)
        correct_sum = sum(predict_AU == AU_targets).sum()
        acc_AU_batch = correct_sum.float()/(bsz*args.AU_cls)
        acc_AU.update(acc_AU_batch, bsz)
        label_AU['gt'].append(AU_targets.cpu().detach().numpy())
        label_AU['pred'].append(predict_AU.cpu().detach().numpy())
        
        #warm up
        if args.warmup>0 and iteration <= warmup_iter :
            warmup_scheduler.step()
            warm_lr = warmup_scheduler.get_lr()
            # print("warm_lr:%s" % round(warm_lr[0],5))

        elif args.scheduler :
            scheduler.step()  # update lr
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if(iteration + 1) % print_freq == 0:
            if iteration <= warmup_iter and args.warmup>0 :
                print('Train: [{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU))
                sys.stdout.flush()

            else:
                print('Train: [{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'
                    'lr {lr:.8f}\t'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR, acc_AU=acc_AU))

        if(iteration + 1) % args.valid_freq == 0:
            # eval
            time3 = time.time()
            loss,EXPR_accs= validate_FER(FER_val_loader,model, criterion[0], args)
            time4 = time.time()
            print('Validation iter {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
                iteration+1, time4 - time3, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2]))
            if args.save_model:
                if EXPR_accs[1]>= 67:
                    save_file = os.path.join(
                        args.save_folder, 'iter_{iter}_{acc}.pth'.format(iter=iteration,acc=round(EXPR_accs[1],5)))
                    save_model(model, optimizer, args, iteration, save_file)

            sys.stdout.flush()

def train_iter_FERAU(FER_train_loader,FER_val_loader,model, criterion, optimizer,warmup_scheduler, all_iterations, warmup_iter,scheduler, args,AU_RATIO=1):
    """one epoch training"""
    print_freq=int(all_iterations/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_AU = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    label_AU = {'gt': [], 'pred': []}
    iter_FER_loader=iter(FER_train_loader)

    end = time.time()

    for iteration in range(all_iterations):
        data_time.update(time.time() - end)
        # load AU data
        try:
          images, FER_targets, AU_targets=next(iter_FER_loader)
        except:
          iter_FER_loader=iter(FER_train_loader)
          images, FER_targets, AU_targets=next(iter_FER_loader)


        # process FER labels
        FER_targets = FER_targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        FER_targets = torch.as_tensor(FER_targets, dtype=torch.int64)
        FER_targets= FER_targets.cuda()
        # process AU labels
        AU_target_arr = np.array(AU_targets,dtype='int32').T
        AU_target_tensor = torch.tensor(AU_target_arr)
        AU_targets = AU_target_tensor.cuda()
        #batch-size

        bsz=len(images)

        # imgs to cuda
        images = images.cuda()
        # model
        output = model(images)

        FER_output = output[0]
        AU_output = output[1]
        #AU_output = nn.Parameter(torch.ones(AU_bsz, 21)).cuda()
        loss_EXPR = criterion[0](FER_output, FER_targets)
        loss_AU = criterion[1](AU_output, AU_targets.float())
        loss = loss_EXPR + AU_RATIO * loss_AU


        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update metric
        losses.update(loss, bsz)
        acc_EXPR_batch = accuracy(FER_output, FER_targets)
        acc_EXPR.update(acc_EXPR_batch[0], bsz)
        label_EXPR['gt'].append(FER_targets.cpu().detach().numpy())
        label_EXPR['pred'].append(FER_output.cpu().detach().numpy())
        predict_AU = torch.sigmoid(AU_output)
        predict_AU = torch.round(predict_AU)
        correct_sum = sum(predict_AU == AU_targets).sum()
        acc_AU_batch = correct_sum.float()/(bsz*args.AU_cls)
        acc_AU.update(acc_AU_batch, bsz)
        label_AU['gt'].append(AU_targets.cpu().detach().numpy())
        label_AU['pred'].append(predict_AU.cpu().detach().numpy())
        
        #warm up
        if args.warmup>0 and iteration <= warmup_iter :
            warmup_scheduler.step()
            # warm_lr = warmup_scheduler.get_lr()
            # print("warm_lr:%s" % round(warm_lr[0],5))

        elif args.scheduler :
            scheduler.step()  # update lr
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if(iteration + 1) % print_freq == 0:
            if iteration <= warmup_iter and args.warmup>0 :
                print('Train: [{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'
                    'Warmup {warmup:.5f}'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses, acc_EXPR=acc_EXPR, acc_AU=acc_AU,warmup=warmup_scheduler.get_lr()[0]))
                sys.stdout.flush()

            else:
                print('Train: [{0}/{1}]\t'
                    #   'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #   'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                    'AccAU {acc_AU.val:.3f} ({acc_AU.avg:.3f})\t'
                    'lr {lr:.8f}\t'.format(
                    iteration + 1, all_iterations,
                    # batch_time=batch_time,data_time=data_time, 
                    loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR, acc_AU=acc_AU))

        if(iteration + 1) % args.valid_freq == 0:
            # eval
            time3 = time.time()
            loss,EXPR_accs= validate_FER(FER_val_loader,model, criterion[0], args)
            time4 = time.time()

            print('Validation iter {}, total time {:.2f}, EXPR F1:{:.4f}, accuracy:{:.4f}, total accuracy:{:.4f}'.format(
                iteration+1, time4 - time3, EXPR_accs[0], EXPR_accs[1], EXPR_accs[2]))
            if args.save_model:
                if EXPR_accs[0]>= 0.67:
                    save_file = os.path.join(
                        args.save_folder, 'iter_{iter}_{acc}.pth'.format(iter=iteration,acc=round(EXPR_accs[0],5)))
                    save_model(model, optimizer, args, iteration, save_file)

            sys.stdout.flush()
        model.train()


def train_FER_simsiam(FER_train_loader, model,classifier, criterion, optimizer,epoch,warmup_epoch, args):
    """one epoch training"""
    print_freq=int(FER_train_loader.__len__()/args.print_num)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}


    end = time.time()

    for idx, (images, targets) in enumerate(FER_train_loader):
        data_time.update(time.time() - end)

        # #warm up
        # if epoch <= warmup_epoch and args.warmup:
        #     warmup_scheduler.step()
        #     warm_lr = warmup_scheduler.get_lr()
        #     # print("warm_lr:%s" % warm_lr)

        targets = targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
        targets = torch.as_tensor(targets, dtype=torch.int64)
        images = images.cuda()
        targets= targets.cuda()

        bsz = targets.shape[0]
        classifier.zero_grad()
        with torch.no_grad():
            feature = model(images)
        # model
        output = classifier(feature)
        loss_EXPR = criterion(output, targets)

        # optimize
        optimizer.zero_grad()
        loss_EXPR.backward()
        optimizer.step()

        # update metric
        losses.update(loss_EXPR, bsz)
        acc_EXPR_batch = accuracy(output, targets)
        acc_EXPR.update(acc_EXPR_batch[0], bsz)
        label_EXPR['gt'].append(targets.cpu().detach().numpy())
        label_EXPR['pred'].append(output.cpu().detach().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if  epoch <= warmup_epoch and args.warmup and (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'.format(
                epoch, idx + 1, len(FER_train_loader),loss=losses, acc_EXPR=acc_EXPR))
                # 'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # 'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #  batch_time=batch_time,data_time=data_time,
            sys.stdout.flush()
        elif (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'
                'lr {lr:.8f}\t'.format(
                epoch, idx + 1, len(FER_train_loader), loss=losses,lr=optimizer.param_groups[0]['lr'], acc_EXPR=acc_EXPR))
            sys.stdout.flush() 


    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs

def validate_FER_simsiam(val_loader, model,classifier, criterion, args):
    """validation"""
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_EXPR = AverageMeter()
    label_EXPR = {'gt': [], 'pred': []}
    with torch.no_grad():
        end = time.time()
        for idx, (images, targets) in enumerate(val_loader):
            targets = targets.squeeze()   # if parameters are none ,tensor is squeezed into one dimension
            targets = torch.as_tensor(targets, dtype=torch.int64)
            images = images.cuda()
            targets= targets.cuda()
  
            # bsz = len(targets)  
            bsz = targets.shape[0]
            feature = model(images)
            output = classifier(feature)
            loss_EXPR = criterion(output, targets)


            # update metric
            losses.update(loss_EXPR, bsz)
            acc_EXPR_batch = accuracy(output, targets)
            acc_EXPR.update(acc_EXPR_batch[0], bsz)
            label_EXPR['gt'].append(targets.cpu().detach().numpy())
            label_EXPR['pred'].append(output.cpu().detach().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if idx % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'AccEXPR {acc_EXPR.val:.3f} ({acc_EXPR.avg:.3f})\t'.format(
            #            idx, len(val_loader), batch_time=batch_time,
            #            loss=losses, acc_EXPR=acc_EXPR))
            #     sys.stdout.flush()

    label_gt = np.concatenate(label_EXPR['gt'], axis=0)
    label_pred = np.concatenate(label_EXPR['pred'], axis=0)
    f1, acc, total_acc = EXPR_metric(label_pred, label_gt)
    EXPR_accs = [f1, acc, total_acc]

    return losses.avg, EXPR_accs


def infer_FER(val_loader, model, args):
    """inference"""
    model.eval()
    batch_time = AverageMeter()
    with torch.no_grad(),open(os.path.join(args.model_path,args.result_file_name),"w") as output_file:
        end = time.time()
        for idx, (images, name) in enumerate(val_loader):
            images = images.cuda()

            # model
            output = model(images)
            # write out
            result=output.cpu().detach().numpy()
            for i in range(len(name)):
                distribut=result[i]
                # cls_index=distribut.index(max(distribut))
                cls_index=np.argmax(distribut)
                content=name[i]+","+str(cls_index)+'\n'
                output_file.write(content)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print proceduce
            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time))
                sys.stdout.flush()
    output_file.close()

def infer_FER_AUViT(val_loader, model, args):
    """inference"""
    model.eval()
    batch_time = AverageMeter()
    with torch.no_grad(),open(os.path.join(args.model_path,args.result_file_name),"w") as output_file:
        end = time.time()
        for idx, (images, name) in enumerate(val_loader):
            images = images.cuda()

            # model
            output = model(images)[0]
            # write out
            result=output.cpu().detach().numpy()
            for i in range(len(name)):
                distribut=result[i]
                # cls_index=distribut.index(max(distribut))
                cls_index=np.argmax(distribut)
                content=name[i]+","+str(cls_index)+'\n'
                output_file.write(content)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print proceduce
            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time))
                sys.stdout.flush()
    output_file.close()




