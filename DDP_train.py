import math

import numpy as np
import network as yv4
import YV4_loss as LF
import torch as tr
from torch.cuda import amp
import torch.optim as opt
import tqdm
import time
import os
import torch.cuda as cuda
import shutil
import cv2
import torch.nn as nn
import torch.multiprocessing as trmp
import dataset_cp as ndata
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import collections

dataset_field = ['class_names',
                     'cluster_anchors',
                     'input_size',
                     'dirs',
                     'xyxy',
                     'label_type',
                     'class_first',
                     'cache_label',
                     'normalize_label',
                     'label_is_normalized',
                     'to_xywh',
                     'read_label_from_cache']
tr_field = ['nc',
                'accum',
                'batch_size',
                'lr',
                'epoch',
                'input_size',
                'model_path',
                'load_model',
                'class_names',
                'n_fold',
                'anchors',
                'dataset']
Dataset_params = collections.namedtuple('Dataset_params', dataset_field)
Train_params = collections.namedtuple('Train_params', tr_field)


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = tr.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with tr.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        datas, labels = self.next_input, self.next_target
        self.preload()
        return datas, labels


def save_model(dict):
    tr.save(dict)


def output_size_producer(input_size, scales):
    return [(input_size[0] // scale, input_size[0] // scale) for scale in scales]


def data_separet(datas, labels, gpus):
    batch_size = datas.shape[0]
    sub_batch = batch_size // gpus
    return (datas[:sub_batch, ...], datas[sub_batch:, ...]), (labels[:sub_batch, ...], labels[sub_batch:, ...])


def train(params):
    ...


def average_grad(net, group):
    size = float(dist.get_world_size())
    for p in net.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, group=group)
            p.grad.data /= size


def run(rank, params, net):
    """ Distributed function to be implemented later. """
    dataset = params.dataset
    world_size = dist.get_world_size()
    device = "cuda:{}".format(rank)
    anchors = params.anchors.to(device=device)
    # anchors.requires_grad = True
    loss_fn = LF.ComputeLoss(anchors)
    train_sampler = tr.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    # data_loader = DLX(dataset, batch_size=params['batch_size'], num_workers=5,
    #                          sampler=train_sampler, collate_fn=ndata.dataset.collate_fn, pin_memory=True)
    data_loader = tr.utils.data.DataLoader(dataset, prefetch_factor=6, batch_size=params.batch_size * 2, num_workers=4,
                                           sampler=train_sampler, collate_fn=ndata.dataset.collate_fn, pin_memory=True)
    # data_loader = data_prefetcher(data_loader)S
    # net = yv4.YOLOV4(params.nc, 3, 32, params.n_fold)
    # net.train()
    net.cuda()
    # net.half()
    # net = tr.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # opt_param = {'params': net.parameters(), 'initial_lr': params['lr']}
    opt_params = [{'params': net.parameters(), 'lr': params.lr}, {'params': anchors, 'lr': params.lr}]
    optimizer = opt.AdamW(opt_params)
    # cos_an = opt.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1, eta_min=1e-6,
    #                                                       last_epoch=params['epoch'] - 1)
    cos_an = opt.lr_scheduler.CosineAnnealingLR(optimizer, params.epoch, eta_min=1e-6, last_epoch=-1)
    if params.load_model:
        ckpt = tr.load(params.model_path, map_location=device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        cos_an.load_state_dict(ckpt['cos_an'])
        # loss_fn.anchors[:, :] = ckpt['anchors'][:, :]
    net = DDP(net, device_ids=[rank])

    resume_e = ckpt['epoch'] if params.load_model else 0

    # scheduler = opt.lr_scheduler.CyclicLR(optimizer, base_lr=params['lr'], max_lr=5e-3, cycle_momentum=False)
    # net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    scaler = amp.GradScaler(enabled=True)
    accum = params.accum
    aug = ndata.Pre_augment(True, params.input_size, params.batch_size)
    for i in range(resume_e, params.epoch):
        train_sampler.set_epoch(i)
        if rank == 0:
            prefix = 'Epoch {}'.format(i)
            data_loader = tqdm.tqdm(data_loader, leave=True, desc=prefix, ncols=150)
        count = 0
        re_l = 0.0
        other_loss = 0.0
        optimizer.zero_grad()
        for datas, labels in data_loader:
            datas = datas.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            datas, labels = aug(datas, labels)
            # print(labels.shape, '+'*8)
            with amp.autocast(enabled=True):
                output = net(datas)
                loss, ol = loss_fn(output, labels, datas)
            other_loss = other_loss + ol.detach() / accum
            loss = loss / accum
            scaler.scale(loss).backward()
            re_l = re_l + loss.detach()
            count += 1
            if count % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()
                optimizer.zero_grad()
                dist.all_reduce(re_l, op=dist.ReduceOp.SUM)
                dist.all_reduce(other_loss, op=dist.ReduceOp.SUM)
                re_l /= world_size
                other_loss /= world_size
                if rank == 0:
                    postfix = ' Loss: {0:>7.5f}, B_los: {1:>7.5f}, O_los: {2:>7.5f}, C_los: {3:>7.5f}'.format(
                        re_l.item() / count, other_loss[0].item() / count, other_loss[1].item() / count,
                        other_loss[2].item() / count
                    )
                    data_loader.set_postfix_str(postfix)
        if rank == 0:
            # print(loss_fn.anchors)
            state_dict = {'epoch': i,
                          'model_state_dict': net.module.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          "cos_an": cos_an.state_dict(),
                          'anchors': loss_fn.anchors,
                          'cls_names': params.class_names}
            tr.save(state_dict, params.model_path)
        cos_an.step()


def init_process(rank, host, port, fn, params, net, world_size=2, backend='nccl'):
    """ Initialize the distributed environment. """
    tr.cuda.set_device(rank)
    # tr.manual_seed(0)
    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, init_method='env://', world_size=world_size)
    # print('=='*8)
    fn(rank, params, net)


if __name__ == '__main__':
    batch_size = 54
    times = 2
    epoch = 250
    learning_rate = 1e-3
    ini_input_size = [900, 900]
    input_size = [608, 608]
    gpus = 2
    im_dir = r'/media/jr/Data/RDD/For_yolov5/images/train'
    label_dir = r'/media/jr/Data/RDD/For_yolov5/labels/train'
    save_path = r'/home/jr/models/AAR2.pt'
    na = 3
    nl = 3
    cla = ['Drogue']
    # cla = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    # cla = ['zero', 'one', 'two']
    # cla = ['car', 'person']
    # cla = ['Fighter', 'Warship', 'Aircraft-carrier', 'B-2', 'V-22', 'Tank', 'Helicopter']

    data_info = {'class_names': cla,
                 'dirs': (label_dir, im_dir),
                 'cluster_anchors': True,
                 'input_size': ini_input_size,
                 'xyxy': False,
                 'label_type': 'yolo_txt',
                 'class_first': True,
                 'cache_label': False,
                 'normalize_label': True,
                 'label_is_normalized': True,
                 'to_xywh': True,
                 'read_label_from_cache': True}

    tr_info = {'nc': len(cla),
               'accum': times,
               'batch_size': batch_size,
               'lr': learning_rate,
               'epoch': epoch,
               'input_size': input_size,
               'model_path': save_path,
               'load_model': True,
               'class_names': cla,
               'n_fold': 3}

    dataset_params = Dataset_params(**data_info)
    dataset = ndata.dataset(dataset_params, cache_imgs=False)
    anchors = tr.as_tensor(dataset.anchors, dtype=tr.float32)

    tr_info['anchors'] = anchors
    tr_info['dataset'] = dataset
    train_params = Train_params(**tr_info)

    # print(train_params)
    # anchors = ut.read_anchor(r'/media/jr/Data/RDD/detect_model/anchors.txt', 3, ', ')
    ramfy = (4, 6, 8)

    net = yv4.YOLOV4(train_params.nc, 3, 32, train_params.n_fold)
    net.train()

    host = 'localhost'
    port = '1234'
    trmp.spawn(init_process, nprocs=gpus, args=(host, port, run, train_params, net))
