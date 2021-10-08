import cv2
import torch as tr
import os
import glob
import time

import torch.backends.cudnn
import tqdm

import MY_TOOLS.Data_process_lib as dpl
from collections import Iterable
import random
import torch.multiprocessing as mp
from itertools import repeat
import numpy as np
import torchvision.transforms as trf
from multiprocessing.pool import ThreadPool
import math


def add_fog(img):
    device = img.device
    (chs, row, col) = img.shape

    h_m = tr.arange(row, device=device).reshape(row, 1).repeat(1, col)
    w_m = tr.arange(col, device=device).reshape(1, col).repeat(row, 1)

    A = 0.5                              # 亮度
    beta = random.uniform(0, 0.25)                  # 雾的浓度
    size = math.sqrt(max(row, col))/random.uniform(1.4, 2.5)  # 雾化尺寸
    w_off = random.randint(-10, 10)
    h_off = random.randint(-10, 10)
    center = (row//3 + h_off, 3*col//7 + w_off)        # 雾化中心

    d = -0.02 * tr.sqrt((h_m-center[0])**2 + (w_m-center[1])**2) + size
    td = tr.exp(-beta*d)[None, :, :]
    # print(td)
    img_f = img*td + A * (1 - td)
    # for j in range(row):
    #     for l in range(col):
    #         d = -0.02 * math.sqrt((j-center[0])**2 + (l-center[1])**2) + size
    #         td = math.exp(-beta * d)
    #         img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    return img_f



class add_rain:
    def __init__(self, denseness, thickness, length, angle, type):
        self.denseness = denseness
        self.thickness = thickness
        self.length = length
        self.min_angle = angle[0]
        self.max_angle = angle[1]
        self.angle = random.randint(*angle) -60
        self.type = type

    def get_noise(self,  img):
        th = random.randint(*self.denseness)
        noise = np.random.uniform(0, 255, img.shape[0:2])
        # noise[np.where(noise > (255 - th))] = 0
        noise[np.where(noise < (230 - th))] = 0
        # im = noise.astype(np.uint8)
        # cv2.imshow('im',im)
        # cv2.waitKey()

        k = np.array([[0, 0.1, 0],
                      [0.1, 8, 0.1],
                      [0, 0.1, 0]])
        noise = cv2.filter2D(noise, -1, k)
        return noise

    def rain(self, noise):
        a_angle = random.randint(-5, 5)
        self.angle += a_angle
        if self.angle > self.max_angle:
            self.angle = self.max_angle
        elif self.angle < self.min_angle:
            self.angle = self.min_angle
        w = random.randint(*self.thickness)
        if w%2==0:
            w += 1
        length = random.randint(*self.length)
        trans = cv2.getRotationMatrix2D((length / 2, length / 2), self.angle , 1 - length / 100.0)
        dig = np.diag(np.ones(length))
        k = cv2.warpAffine(dig, trans, (length, length))
        k = cv2.GaussianBlur(k, (w, w), 0)
        rain_img = cv2.filter2D(noise, -1, k)
        cv2.normalize(rain_img, rain_img, 0, 255, cv2.NORM_MINMAX)
        rain_img = np.array(rain_img, dtype=np.uint8)
        return rain_img

    def __call__(self, img):
        nois = self.get_noise(img)
        rain_img = self.rain(nois)[:, :, None]
        if self.type == 'normal':
            rain_weight = rain_img/255.0
            img_weight = 1 - rain_weight
            rain_img = rain_img.repeat(3, 2)
            # print(rain_img.shape)
            re_img = rain_img * rain_weight + img * img_weight
        elif self.type == 'smooth':
            ain_img = rain_img.repeat(3, 2)
            re_img = cv2.addWeighted(img, 0.85, ain_img, 1 - 0.85, 1)
        return re_img.astype(np.uint8)


def setup_seed():
    tr.manual_seed(int(math.modf(time.time())[0]*100))
    tr.cuda.manual_seed(int(math.modf(time.time())[0]*100))
    random.seed(int(math.modf(time.time())[0]*100))
    torch.backends.cudnn.deterministic = True



def resize_img(img, label, out_size):
    re_img = np.zeros((*out_size, img.shape[2]), dtype=img.dtype)
    # print(re_img.shape)
    h, w, _ = img.shape
    hr, wr, = out_size[0] / h, out_size[1] / w
    if hr <= wr:
        img = cv2.resize(img, (0, 0), fx=hr, fy=hr)
        h, w, _ = img.shape
        margin = out_size[1] - w
        offset = random.randint(0, margin)
        label[:, 1] = label[:, 1] * w / out_size[1] + (offset / out_size[1])
        label[:, 3] = label[:, 3] * w / out_size[1]
        re_img[:h, offset: offset + w, :] = img

    else:
        img = cv2.resize(img, (0, 0), fx=wr, fy=wr)
        h, w, _ = img.shape
        margin = out_size[0] - h
        offset = random.randint(0, margin)
        label[:, 2] = label[:, 2] * h / out_size[0] + (offset / out_size[0])
        label[:, 4] = label[:, 4] * h / out_size[0]
        re_img[offset: offset + h, :w, :] = img
    # print(label)
    return re_img, label


def draw_box(im, boxes, classes=None, col=(0, 0, 255)):
    def draw_single_box(box):
        # print(box)
        box = box.astype(int)
        # print(im.shape, im.dtype, box[:4], col)
        xmin, ymin, xmax, ymax = box[:4]
        temp_im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), col, 1)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # text = classes[box[4] - 1]
        # cv2.putText(im, text, (box[0], box[1]-5), font, 0.8, col, 1)
        return temp_im

    for box in boxes:
        im = draw_single_box(box)
    return im


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, tr.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, tr.Tensor) else np.copy(x)
    y[:, (0, 1)] = x[:, (0, 1)] - (x[:, (2, 3)] / 2)
    y[:, (2, 3)] = x[:, (0, 1)] + (x[:, (2, 3)] / 2)
    return y


def data_enhance(img, labels, enhance_type, random_pram=None, pram=None):
    # bbs = BoundingBoxesOnImage([BoundingBox(*v) for v in label], img.shape)
    if enhance_type == 'flip':
        tmp = random.randint(0, 2)
        if tmp == 0:
            lr_img = np.fliplr(img)
            labels[:, 1] = 2 * (0.5 - labels[:, 1]) + labels[:, 1]
            return lr_img, labels
        elif tmp == 1:
            ud_img = np.flipud(img)
            labels[:, 2] = 2 * (0.5 - labels[:, 2]) + labels[:, 2]
            return ud_img, labels
        else:
            img = np.fliplr(img)
            labels[:, 1] = 2 * (0.5 - labels[:, 1]) + labels[:, 1]

            lrup_img = np.flipud(img)
            labels[:, 2] = 2 * (0.5 - labels[:, 2]) + labels[:, 2]
            return lrup_img, labels
    if enhance_type == 'gama':
        lens = len(random_pram)
        p = random.randint(0, lens - 1)
        temp_im = img[:, :, :]
        gama_img = temp_im ** random_pram[p]
        return gama_img, labels


def load_img(pld):
    pl, device = pld
    path, l = pl
    img = cv2.imread(path)
    img = tr.as_tensor(img, device=device).permute(2, 0, 1)/255.0
    _, h, w = img.shape
    label = l.detach()
    label[:, (1, 3)] = label[:, (1, 3)] * w
    label[:, (2, 4)] = label[:, (2, 4)] * h
    return img, label


class Pre_augment(tr.nn.Module):
    def __init__(self, aug_first, img_sze, batch_size, mosaic=True, nf=None):
        super(Pre_augment, self).__init__()
        self.img_size = img_sze
        self.batch_size = batch_size
        self.aug_first = aug_first
        self.mos = mosaic
        self.pix_aug = trf.Compose([
            trf.RandomAdjustSharpness(2),
            trf.RandomAutocontrast(),
            trf.ColorJitter((0.5, 1.4), (0.5, 1.4), (0.4, 1.5)),
            add_fog
        ])

    def _random_affine(self, img, labels, min_rotd=-18, max_rotd=18, min_sd=-10, max_sd=10):
        device = img.device
        fill = random.uniform(0, 1)
        rot_d = random.randint(min_rotd, max_rotd)
        degrees = [90, -90, 180, -180, rot_d]
        rot_d = random.choice(degrees)
        if random.randint(0, 21) % 2 == 0:
            shear_x = random.randint(min_sd, max_sd)
            shear = (-shear_x, -shear_x)
            S_rad = shear_x * (math.pi / 180)
            S_M = tr.tensor([[1, math.tan(S_rad)], [0, 1]], dtype=tr.float32, device=device)
        else:
            shear_y = random.randint(min_sd, max_sd)
            shear = (0, 0, -shear_y, -shear_y)
            S_rad = shear_y * (math.pi / 180)
            S_M = tr.tensor([[1, 0], [math.tan(S_rad), 1]], dtype=tr.float32, device=device)
        rot_fn = trf.transforms.RandomAffine((rot_d, rot_d), shear=shear, fill=fill)
        rot_rad = -rot_d * (math.pi / 180)
        img = rot_fn(img)
        M = tr.tensor([[math.cos(rot_rad), math.sin(rot_rad)],
                       [-math.sin(rot_rad), math.cos(rot_rad)]], dtype=tr.float32, device=device)
        M = tr.matmul(M, S_M)
        im_shape = img.shape
        # print(im_shape, '='*10)
        labels = tr.cat((labels[:, 0:2], labels[:, 2:3], labels[:, 1:2],
                         labels[:, 2:4], labels[:, 0:1], labels[:, 3:4]), dim=-1)
        labels[:, ::2] = labels[:, ::2] - im_shape[-1] // 2
        labels[:, 1::2] = labels[:, 1::2] - im_shape[-2] // 2
        labels = labels.view((labels.shape[0], -1, 2))
        labels = tr.matmul(labels, M.T).view((labels.shape[0], 8))
        labels[..., 0::2] = labels[..., 0::2] + im_shape[-1] // 2
        labels[..., 1::2] = labels[..., 1::2] + im_shape[-2] // 2

        labels = tr.cat((tr.min(labels[:, ::2], -1)[0][:, None],
                         tr.min(labels[:, 1::2], -1)[0][:, None],
                         tr.max(labels[:, ::2], -1)[0][:, None],
                         tr.max(labels[:, 1::2], -1)[0][:, None]),
                        dim=-1)
        labels[..., ::2] = labels[..., ::2].clamp(0, im_shape[-1])
        labels[..., 1::2] = labels[..., 1::2].clamp(0, im_shape[-2])

        return img, labels, labels

    def _augment(self, img, labels):
        img, labels, _ = self._random_affine(img, labels)
        img = self.pix_aug(img)

        # img1 = (img.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
        # img1 = np.ascontiguousarray(img1)
        # print(img1.shape)
        # labels1 = labels.detach().cpu().numpy()
        # h, w, _ = img1.shape
        # tmp_label = np.zeros((labels1.shape[0], 4), dtype=np.float32)
        # tmp_label[:, (0, 2)] = labels1[:, (0, 2)]
        # tmp_label[:, (1, 3)] = labels1[:, (1, 3)]
        # # tmp_label = xywh2xyxy(tmp_label[:, :4])
        # tmp_label = tmp_label.astype(np.int)
        # # print(tmp_label, labels)
        # # img = (img*255).astype(np.uint8)
        # img1 = draw_box(img1, tmp_label)
        # # print(img.shape)
        # cv2.imshow('im', img1)
        # cv2.waitKey()

        return img, labels

    def augment(self, img, labels):
        _, h, w = img.shape
        img, tmp = self._augment(img, labels[:, 2:])
        # print(tmp)
        labels[:, 2:] = tmp
        return img, labels

    def mosiac(self, imgs, labels, lim_pix=40):
        device = imgs.device
        def merg_labels(label, lxy, mxy, thr=0.5, xyxy=True):  # xyxy
            if not xyxy:
                label[:, 2:] = xywh2xyxy(label[:, 2:])
            l_x1, l_x2 = label[:, 2] - mxy[0], label[:, 4] - mxy[0]
            r_x1, r_x2 = label[:, 2] - mxy[2], label[:, 4] - mxy[2]
            u_y1, u_y2 = label[:, 3] - mxy[1], label[:, 5] - mxy[1]
            b_y1, b_y2 = label[:, 3] - mxy[3], label[:, 5] - mxy[3]
            w = (label[:, 4] - label[:, 2]).clamp(1e-6)
            h = (label[:, 5] - label[:, 3]).clamp(1e-6)
            # tr.max()
            l_mask = (l_x1 < 0) & (l_x2 > 0)
            r_mask = (r_x1 < 0) & (r_x2 > 0)
            u_mask = (u_y1 < 0) & (u_y2 > 0)
            b_mask = (b_y1 < 0) & (b_y2 > 0)
            norm_mask = (l_x1 >= 0) & (r_x2 <= 0) & (u_y1 >= 0) & (b_y2 <= 0)

            # a_mask = (l_x1/w < -thr) | (u_y1/h < -thr) | (b_y2/h > thr) | (r_x2/w > thr)
            margin = (l_x1 / w).clamp(max=0) + (u_y1 / h).clamp(max=0) + (-b_y2 / h).clamp(max=0) + (
                    -r_x2 / w).clamp(max=0)
            a_mask = margin < -thr

            mask = (l_mask | r_mask | u_mask | b_mask) & ~a_mask
            mask = mask | norm_mask
            label = label[mask]
            label[:, (2, 4)] = label[:, (2, 4)] - mxy[0]
            label[:, (3, 5)] = label[:, (3, 5)] - mxy[1]
            label[:, 2] = tr.clamp(label[:, 2], 0, mxy[2] - mxy[0])
            label[:, 4] = tr.clamp(label[:, 4], 0, mxy[2] - mxy[0])
            label[:, 3] = tr.clamp(label[:, 3], 0, mxy[3] - mxy[1])
            label[:, 5] = tr.clamp(label[:, 5], 0, mxy[3] - mxy[1])
            label[:, (2, 4)] = label[:, (2, 4)] + lxy[0]
            label[:, (3, 5)] = label[:, (3, 5)] + lxy[1]
            # label[:, 1:] = xyxy2xywh(label[:, 1:])
            return label

        im_ls = []
        for i, img in enumerate(imgs):
            label = labels[i]
            img_shape = img.shape

            # u_r = min(self.img_size[1] / img_shape[2], self.img_size[0] / img_shape[1])
            u_r = 1.0
            w_pix, h_pix = (label[:, -2:].T) * u_r
            if tr.min(h_pix) >= lim_pix and tr.min(w_pix) >= lim_pix:
                min_wr = tr.min(lim_pix / w_pix)
                min_hr = tr.min(lim_pix / h_pix)
                # print(w_pix, tr.min(lim_pix / w_pix), '+++' * 10)
                min_r = min(min_hr, min_wr)
                r = random.uniform(min_r, u_r)
                resize = trf.Resize((int(img_shape[1] * r), int(img_shape[2] * r)))
                img = resize(img)
                label[:, 2:] = label[:, 2:] * r
                if self.aug_first:
                    img, label = self.augment(img, label)
                im_ls.append((img, label))
            else:
                r = u_r
                resize = trf.Resize((int(img_shape[1] * r), int(img_shape[2] * r)))
                img = resize(img)
                label[:, 2:] = label[:, 2:] * r
                if self.aug_first:
                    img, label = self.augment(img, label)
                im_ls.append((img, label))

        d = random.uniform(0, 1)
        re_img = tr.full((3, *self.img_size), d, device=device)
        if imgs.shape[0] == 4:
            xcor, ycor = random.randint(int(0.2 * self.img_size[1]), int(0.8 * self.img_size[1])), \
                         random.randint(int(0.2 * self.img_size[0]), int(0.8 * self.img_size[0]))
            label_list = []
            for i, (img, label) in enumerate(im_ls):
                img_shape = img.shape
                if i == 0:
                    cw, ch = min(xcor, img_shape[2]), min(ycor, img_shape[1])
                    ax1, ay1 = xcor - cw, ycor - ch
                elif i == 1:
                    cw, ch = min((self.img_size[1] - xcor), img_shape[2]), min(ycor, img_shape[1])
                    ax1, ay1 = xcor, ycor - ch
                elif i == 2:
                    cw, ch = min(xcor, img_shape[2]), min((self.img_size[0] - ycor), img_shape[1])
                    ax1, ay1 = xcor - cw, ycor
                else:
                    cw, ch = min((self.img_size[1] - xcor), img_shape[2]), min((self.img_size[0] - ycor), img_shape[1])
                    ax1, ay1 = xcor, ycor
                ax2, ay2 = ax1 + cw, ay1 + ch
                bx1, by1 = random.randint(0, img_shape[2] - cw), random.randint(0, img_shape[1] - ch)
                bx2, by2 = bx1 + cw, by1 + ch

                # print(i)
                re_img[:, ay1: ay2, ax1: ax2] = img[:, by1: by2, bx1: bx2]
                del img
                # label[:, 1:] = xywh2xyxy(label[:, 1:])
                lxy = [ax1, ay1, ax2, ay2]
                mxy = [bx1, by1, bx2, by2]
                label_list.append(merg_labels(label, lxy, mxy))
            labels = tr.cat(label_list, dim=0)

            # img1 = (re_img.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
            # img1 = np.ascontiguousarray(img1)
            # print(img1.shape)
            # labels1 = labels.detach().cpu().numpy()
            # h, w, _ = img1.shape
            # tmp_label = np.zeros((labels1.shape[0], 4), dtype=np.float32)
            # tmp_label[:, (0, 2)] = labels1[:, (2, 4)]
            # tmp_label[:, (1, 3)] = labels1[:, (3, 5)]
            # # tmp_label = xywh2xyxy(tmp_label[:, :4])
            # tmp_label = tmp_label.astype(np.int)
            # # print(tmp_label, labels)
            # # img = (img*255).astype(np.uint8)
            # img1 = draw_box(img1, tmp_label)
            # # print(img.shape)
            # cv2.imshow('im', img1)
            # cv2.waitKey()

        return re_img, labels

    def sigale_im(self, ids):
        ims = self.ims[ids]
        labels = []
        for id in ids:
            mask = self.labels[..., 0] == id
            labels.append(self.labels[mask])
        return self.mosiac(ims, labels)

    def forward(self, ims, labels):
        _, h, w, _ = ims.shape
        nd = ims.shape[0]
        self.ims = ims.permute(0, 3, 1, 2)/255.0
        labels[:, (2, 4)] = labels[:, (2, 4)] * w
        labels[:, (3, 5)] = labels[:, (3, 5)] * h
        labels[:, 2:] = xywh2xyxy(labels[:, 2:])
        self.labels = labels
        ids = list(range(nd))
        setup_seed()
        nids = [random.sample(ids, 4) for i in range(self.batch_size)]
        # print(nids[0])
        results = list(ThreadPool(self.batch_size).imap(self.sigale_im, nids))
        labels = []
        imgs = [r for r, _ in results]
        imgs = tr.stack(imgs, dim=0)
        # print(results[0][0].shape, imgs.shape)
        for i, (_, l) in enumerate(results):
            l[:, 0] = i
            labels.append(l)
        labels = tr.cat(labels, dim=0)
        labels[:, (2, 4)] = labels[:, (2, 4)] / self.img_size[1]
        labels[:, (3, 5)] = labels[:, (3, 5)] / self.img_size[0]
        labels[:, 2:] = xyxy2xywh(labels[:, 2:])
        return imgs, labels


class Augment:
    def __init__(self, aug_first, aug_fn, mosaic=True, nf=None):
        self.aug_fn = aug_fn
        if self.aug_fn:
            rm_type = []
            enh_type = self.aug_fn.keys()
            for name in enh_type:
                if name not in ['flip', 'gama', 'mosic', 'mxup']:
                    del self.aug_fn[name]
            if len(rm_type):
                print("Warning: The following enhancement will be removed from "
                      "enhancement list due to that the enhance type doesn't contained.")
                print('Removed enhancement types:' + ' '.join(rm_type))
        self.aug_key = list(self.aug_fn.keys())
        self.nf = nf if nf else len(self.aug_key)
        self.aug_first = aug_first
        self.mos = mosaic
        self.pix_aug = trf.Compose([
            trf.RandomAdjustSharpness(2),
            trf.RandomAutocontrast(),
            trf.ColorJitter((0.4, 1.5), (0.4, 1.5), (0.4, 2))
        ])

    def _random_affine(self, img, labels, min_rotd=-180, max_rotd=180, min_sd=-20, max_sd=20):
        device = img.device
        fill = random.uniform(0, 1)
        rot_d = random.randint(min_rotd, max_rotd)
        if random.randint(0, 21) % 2 == 0:
            shear_x = random.randint(min_sd, max_sd)
            shear = (-shear_x, -shear_x)
            S_rad = shear_x * (math.pi / 180)
            S_M = tr.tensor([[1, math.tan(S_rad)], [0, 1]], dtype=tr.float32, device=device)
        else:
            shear_y = random.randint(min_sd, max_sd)
            shear = (0, 0, -shear_y, -shear_y)
            S_rad = shear_y * (math.pi / 180)
            S_M = tr.tensor([[1, 0], [math.tan(S_rad), 1]], dtype=tr.float32, device=device)
        rot_fn = trf.transforms.RandomAffine((rot_d, rot_d), shear=shear, fill=fill)
        rot_rad = -rot_d * (math.pi / 180)
        img = rot_fn(img)
        M = tr.tensor([[math.cos(rot_rad), math.sin(rot_rad)],
                       [-math.sin(rot_rad), math.cos(rot_rad)]], dtype=tr.float32, device=device)
        M = tr.matmul(M, S_M)
        im_shape = img.shape
        # print(im_shape, '='*10)
        labels = tr.cat((labels[:, 0:2], labels[:, 2:3], labels[:, 1:2],
                         labels[:, 2:4], labels[:, 0:1], labels[:, 3:4]), dim=-1)
        labels[:, ::2] = labels[:, ::2] - im_shape[-1] // 2
        labels[:, 1::2] = labels[:, 1::2] - im_shape[-2] // 2
        labels = labels.view((labels.shape[0], -1, 2))
        labels = tr.matmul(labels, M.T).view((labels.shape[0], 8))
        labels[..., 0::2] = labels[..., 0::2] + im_shape[-1] // 2
        labels[..., 1::2] = labels[..., 1::2] + im_shape[-2] // 2

        labels = tr.cat((tr.min(labels[:, ::2], -1)[0][:, None],
                         tr.min(labels[:, 1::2], -1)[0][:, None],
                         tr.max(labels[:, ::2], -1)[0][:, None],
                         tr.max(labels[:, 1::2], -1)[0][:, None]),
                        dim=-1)
        labels[..., ::2] = labels[..., ::2].clamp(0, im_shape[-1])
        labels[..., 1::2] = labels[..., 1::2].clamp(0, im_shape[-2])

        return img, labels, labels

    def _augment(self, img, labels, enhance_type, random_pram=None, pram=None):
        img, labels, _ = self._random_affine(img, labels)
        img = self.pix_aug(img)

        # img1 = (img.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
        # img1 = np.ascontiguousarray(img1)
        # print(img1.shape)
        # labels1 = labels.detach().cpu().numpy()
        # h, w, _ = img1.shape
        # tmp_label = np.zeros((labels1.shape[0], 4), dtype=np.float32)
        # tmp_label[:, (0, 2)] = labels1[:, (0, 2)]
        # tmp_label[:, (1, 3)] = labels1[:, (1, 3)]
        # # tmp_label = xywh2xyxy(tmp_label[:, :4])
        # tmp_label = tmp_label.astype(np.int)
        # # print(tmp_label, labels)
        # # img = (img*255).astype(np.uint8)
        # img1 = draw_box(img1, tmp_label)
        # # print(img.shape)
        # cv2.imshow('im', img1)
        # cv2.waitKey()

        return img, labels

    def augment(self, img, labels, device):
        aug = random.sample(self.aug_key, self.nf)
        _, h, w = img.shape
        for a in aug:
            img, tmp = self._augment(img, labels[:, 1:], a, self.aug_fn[a])
            # print(tmp)
            labels[:, 1:] = tmp
        return img, labels

    def mosiac(self, datas, img_szes, device, lim_pix=20):

        def merg_labels(label, lxy, mxy, thr=0.5, xyxy=True):  # xyxy
            if not xyxy:
                label[:, 1:] = xywh2xyxy(label[:, 1:])
            l_x1, l_x2 = label[:, 1] - mxy[0], label[:, 3] - mxy[0]
            r_x1, r_x2 = label[:, 1] - mxy[2], label[:, 3] - mxy[2]
            u_y1, u_y2 = label[:, 2] - mxy[1], label[:, 4] - mxy[1]
            b_y1, b_y2 = label[:, 2] - mxy[3], label[:, 4] - mxy[3]
            w = (label[:, 3] - label[:, 1]).clamp(1e-6)
            h = (label[:, 4] - label[:, 2]).clamp(1e-6)
            # tr.max()
            l_mask = (l_x1 < 0) & (l_x2 > 0)
            r_mask = (r_x1 < 0) & (r_x2 > 0)
            u_mask = (u_y1 < 0) & (u_y2 > 0)
            b_mask = (b_y1 < 0) & (b_y2 > 0)
            norm_mask = (l_x1 >= 0) & (r_x2 <= 0) & (u_y1 >= 0) & (b_y2 <= 0)

            # a_mask = (l_x1/w < -thr) | (u_y1/h < -thr) | (b_y2/h > thr) | (r_x2/w > thr)
            margin = (l_x1 / w).clamp(max=0) + (u_y1 / h).clamp(max=0) + (-b_y2 / h).clamp(max=0) + (
                    -r_x2 / w).clamp(max=0)
            a_mask = margin < -thr

            mask = (l_mask | r_mask | u_mask | b_mask) & ~a_mask
            mask = mask | norm_mask
            label = label[mask]
            label[:, (1, 3)] = label[:, (1, 3)] - mxy[0]
            label[:, (2, 4)] = label[:, (2, 4)] - mxy[1]
            label[:, 1] = tr.clamp(label[:, 1], 0, mxy[2] - mxy[0])
            label[:, 3] = tr.clamp(label[:, 3], 0, mxy[2] - mxy[0])
            label[:, 2] = tr.clamp(label[:, 2], 0, mxy[3] - mxy[1])
            label[:, 4] = tr.clamp(label[:, 4], 0, mxy[3] - mxy[1])
            label[:, (1, 3)] = label[:, (1, 3)] + lxy[0]
            label[:, (2, 4)] = label[:, (2, 4)] + lxy[1]
            # label[:, 1:] = xyxy2xywh(label[:, 1:])
            return label

        im_ls = []
        for img, label in datas:
            img_shape = img.shape

            u_r = min(img_szes[1] / img_shape[2], img_szes[0] / img_shape[1])
            w_pix, h_pix = (label[:, -2:].T) * u_r
            if tr.min(h_pix) >= lim_pix and tr.min(w_pix) >= lim_pix:
                min_wr = tr.min(lim_pix / w_pix)
                min_hr = tr.min(lim_pix / h_pix)
                min_r = min(min_hr, min_wr)
                r = random.uniform(min_r, u_r)
                resize = trf.Resize((int(img_shape[1] * r), int(img_shape[2] * r)))
                img = resize(img)
                label[:, 1:] = label[:, 1:] * r
                label[:, 1:] = xywh2xyxy(label[:, 1:])
                if self.aug_first:
                    img, label = self.augment(img, label, device)
                im_ls.append((img, label))
            else:
                r = u_r
                resize = trf.Resize((int(img_shape[1] * r), int(img_shape[2] * r)))
                img = resize(img)
                label[:, 1:] = label[:, 1:] * r
                label[:, 1:] = xywh2xyxy(label[:, 1:])
                if self.aug_first:
                    img, label = self.augment(img, label, device)
                im_ls.append((img, label))

        d = random.uniform(0, 1)
        re_img = tr.full((3, *img_szes), d, device=device)
        if len(datas) == 4:
            xcor, ycor = random.randint(int(0.2 * img_szes[1]), int(0.8 * img_szes[1])), \
                         random.randint(int(0.2 * img_szes[0]), int(0.8 * img_szes[0]))
            label_list = []
            for i, (img, label) in enumerate(im_ls):
                img_shape = img.shape
                if i == 0:
                    cw, ch = min(xcor, img_shape[2]), min(ycor, img_shape[1])
                    ax1, ay1 = xcor - cw, ycor - ch
                elif i == 1:
                    cw, ch = min((img_szes[1] - xcor), img_shape[2]), min(ycor, img_shape[1])
                    ax1, ay1 = xcor, ycor - ch
                elif i == 2:
                    cw, ch = min(xcor, img_shape[2]), min((img_szes[0] - ycor), img_shape[1])
                    ax1, ay1 = xcor - cw, ycor
                else:
                    cw, ch = min((img_szes[1] - xcor), img_shape[2]), min((img_szes[0] - ycor), img_shape[1])
                    ax1, ay1 = xcor, ycor
                ax2, ay2 = ax1 + cw, ay1 + ch
                bx1, by1 = random.randint(0, img_shape[2] - cw), random.randint(0, img_shape[1] - ch)
                bx2, by2 = bx1 + cw, by1 + ch

                # print(i)
                re_img[:, ay1: ay2, ax1: ax2] = img[:, by1: by2, bx1: bx2]
                # label[:, 1:] = xywh2xyxy(label[:, 1:])
                lxy = [ax1, ay1, ax2, ay2]
                mxy = [bx1, by1, bx2, by2]
                label_list.append(merg_labels(label, lxy, mxy))
            labels = tr.cat(label_list, dim=0)
            labels[:, 1:] = xyxy2xywh(labels[:, 1:])

            # tmp_label = np.zeros((labels.shape[0], 4), dtype=np.float32)
            # tmp_label[:, (0, 2)] = labels[:, (1, 3)]
            # tmp_label[:, (1, 3)] = labels[:, (2, 4)]
            # tmp_label = xywh2xyxy(tmp_label[:, :4])
            # tmp_label = tmp_label.astype(np.int)
            # print(tmp_label, labels)
            # re_img = (re_img * 255).astype(np.uint8)
            # re_img = draw_box(re_img, tmp_label)
            # cv2.imshow('im', re_img)
            # cv2.waitKey()
            # labels[:, (1, 3)] = labels[:, (1, 3)]/img_szes[1]
            # labels[:, (2, 4)] = labels[:, (2, 4)]/img_szes[0]
        return re_img, labels

    def __call__(self, im_path_label, im_sze, device):
        if self.mos:
            if isinstance(im_path_label, list) and len(im_path_label) == 4:
                datas = []
                for path, l in im_path_label:
                    if not isinstance(path, np.ndarray):
                        tmp_im = tr.as_tensor(cv2.imread(path), device=device).permute(2, 0, 1)/255.0
                    else:
                        tmp_im = tr.as_tensor(path, device=device).permute(2, 0, 1)/255.0
                    _, h, w = tmp_im.shape
                    label = l.detach()
                    label[:, (1, 3)] = label[:, (1, 3)] * w
                    label[:, (2, 4)] = label[:, (2, 4)] * h
                    datas.append((tmp_im, label))
                img, labels = self.mosiac(datas, im_sze, device)
            else:
                raise Exception('The images path is not enough or type wrong when using mosaic augmentation')
        else:
            if not isinstance(im_path_label[0], np.ndarray):
                tmp_im = tr.as_tensor(cv2.imread(im_path_label[0]), device=device).permute(2, 0, 1) / 255.0
            else:
                tmp_im = im_path_label[0]
            label = im_path_label[1].detach()
            _, h, w = tmp_im.shape
            label[:, (1, 3)] = label[:, (1, 3)] * w
            label[:, (2, 4)] = label[:, (2, 4)] * h
            label[:, 1:] = xywh2xyxy(label[:, 1:])
            img, labels = self.augment(tmp_im, label)
            label[:, 1:] = xyxy2xywh(label[:, 1:])
        return img, labels


class dataset(tr.utils.data.Dataset):
    def __init__(self, dataset_info, cache_imgs=False, na=3, ns=3, prosc_fn=None, augment=None):
        self.dataset_info = dataset_info
        self.normalize_label = dataset_info.normalize_label
        self.class_first = dataset_info.class_first
        self.xyxy = dataset_info.xyxy
        self.cache_imgs = cache_imgs
        self.clu_anchors = dataset_info.cluster_anchors
        self.to_xywh = dataset_info.to_xywh
        self.na = na
        self.ns = ns
        self.augment = augment
        self.class_names = dataset_info.class_names
        label_dir = dataset_info.dirs[0]
        img_dir = dataset_info.dirs[1]
        if augment:
            if 'mosaic' in augment.keys():
                self.mosaic = augment['mosaic']
                del augment['mosaic']
            else:
                self.mosaic = False
            self.augment = Augment(True, augment, nf=2, mosaic=self.mosaic)
        self.label_type = dataset_info.label_type
        self.img_size = dataset_info.input_size
        self.read_l_from_cache = dataset_info.read_label_from_cache
        self.check_label_info()
        if self.label_type == 'xml':
            if not hasattr(dataset_info, 'class_names'):
                raise Exception('You should provide "class names" when the "label type" is xml')
            glob_name = os.path.join(label_dir, '*.xml')
            label_paths = glob.glob(glob_name)
        elif self.label_type == 'txt':
            glob_name = os.path.join(label_dir, '*.txt')
            label_paths = glob.glob(glob_name)
        else:
            glob_name = os.path.join(label_dir, '*.txt')
            label_paths = glob.glob(glob_name)
        img_format = ['jpg', 'png', 'tiff', 'bmp']

        self.paths = []
        for fm in img_format:
            self.paths.extend(
                [(l, os.path.join(os.path.join(img_dir, os.path.basename(l)[:-4] + '.' + fm))) for l in label_paths
                 if os.path.exists(os.path.join(img_dir, os.path.basename(l)[:-4] + '.' + fm))])

        if dataset_info.cache_label:
            self.cache_label()
        if self.read_l_from_cache:
            cache_path = os.path.join(os.path.dirname(label_dir), 'cache')
            # print(cache_path)
            if not os.path.exists(cache_path):
                print('The cache file is missing.')
                self.cache_label()
            else:
                self.datas = self.read_caching_label(cache_path)
                if len(self.datas) != len(self.paths):
                    self.paths = [(l, i) for l, i in self.paths if i in self.datas.keys()]
                    print("Waring: The cached data doesn't match the non-cache data. "
                          "The cached data lens is {0} and the non-cached data lens is {1}"
                          .format(len(self.datas), len(self.paths)))
        if cache_imgs:
            self.cache_all_img()
        self.prosc_fn = prosc_fn if prosc_fn else None

    def check_label_info(self):
        if self.label_type == 'xml':
            self.l_is_normalized = False
        elif self.label_type == 'txt':
            self.l_is_normalized = self.dataset_info.label_is_normalized
        else:
            self.l_is_normalized = True

    def label_cluster(self, labels):
        from sklearn.cluster import KMeans
        if labels.shape[1] != 2:
            raise Exception('The labels second dimension has to be 2, when clustering anchors.')
        # labels = labels*self.img_size
        km = KMeans(n_clusters=self.na * self.ns, init='k-means++', max_iter=1000, precompute_distances=True)
        km.fit(labels)
        cluter_centers = km.cluster_centers_
        cluter_centers = np.sort(cluter_centers, axis=0)
        # print(cluter_centers)
        return cluter_centers.reshape((self.ns, self.na, -1))

    def cache_label(self):
        # print(self.paths)
        with mp.Pool(4) as pool:
            pbar = list(tqdm.tqdm(pool.imap_unordered(self.load_labels, self.paths),
                             desc='Caching Labels', total=len(self.paths), ncols=100))
            # pbar = pool.imap_unordered(self.load_labels, self.paths)
            self.datas = {p[1]: p[0] for p in pbar}
            clu_labels = [p[2] for p in pbar]

        if self.clu_anchors:
            # print(clu_labels)
            # labels = list(self.datas.values())
            labels = np.concatenate(clu_labels, axis=0)
            if self.class_first:
                self.anchors = self.label_cluster(labels[:, 3:])
            else:
                self.anchors = self.label_cluster(labels[:, 2:4])
        import pickle
        if self.clu_anchors:
            self.datas['anchors'] = self.anchors
        parent = os.path.dirname(os.path.dirname(self.paths[0][0]))
        f_name = os.path.join(parent, 'cache')
        print('Caching to local disk ' + f_name)
        with open(f_name, 'wb') as f:
            pickle.dump(self.datas, f)

    def read_caching_label(self, path):
        import pickle
        with open(path, 'rb') as f:
            labels = pickle.load(f)
        if 'anchors' in labels.keys():
            self.anchors = labels['anchors']
            del labels['anchors']
        else:
            raise Exception('Anchor is missing in cached label file.')

        return labels

    def load_labels(self, paths):
        path = paths[0]
        im_path = paths[1]
        # print('=='*10)
        if self.label_type == 'xml':
            label = dpl.readlables(path, self.class_names, with_onehot=False)
            if self.to_xywh:
                if self.xyxy and self.class_first:
                    label[:, 1:] = xyxy2xywh(label[:, 1:])
                elif self.xyxy and not self.class_first:
                    label[:, :-1] = xyxy2xywh(label[:, :-1])
            else:
                if not self.xyxy and self.class_first:
                    label[:, 1:] = xywh2xyxy(label[:, 1:])
                elif not self.xyxy and not self.class_first:
                    label[:, :-1] = xywh2xyxy(label[:, :-1])
        elif self.label_type == 'yolo_type':
            label = dpl.readlables(path, len(self.class_names), self.label_type)
        else:
            label = dpl.readlables(path, len(self.class_names), self.label_type)
            if self.to_xywh:
                if self.xyxy and self.class_first:
                    label[:, 1:] = xyxy2xywh(label[:, 1:])
                elif self.xyxy and not self.class_first:
                    label[:, :-1] = xyxy2xywh(label[:, :-1])
            else:
                if not self.xyxy and self.class_first:
                    label[:, 1:] = xywh2xyxy(label[:, 1:])
                elif not self.xyxy and not self.class_first:
                    label[:, :-1] = xywh2xyxy(label[:, :-1])

        # print('++'*10)
        label = label.astype(np.float32)
        if self.class_first:
            if np.max(label[:, 1:]) <= 1:
                self.l_is_normalized = True
            else:
                self.l_is_normalized = False
        else:
            if np.max(label[:, :-1]) <= 1:
                self.l_is_normalized = True
            else:
                self.l_is_normalized = False

        if not self.class_first:
            new_label = np.zeros_like(label)
            new_label[:, 0] = label[:, -1]
            new_label[:, 1:] = label[:, :-1]
            label = new_label

        # img = cv2.imread(im_path)
        # h, w, _ = img.shape
        # tmp_label = np.zeros((label.shape[0], 4), dtype=np.float32)
        # tmp_label[:, (0, 2)] = label[:, (1, 3)] * w
        # tmp_label[:, (1, 3)] = label[:, (2, 4)] * h
        # tmp_label = xywh2xyxy(tmp_label[:, :4])
        # tmp_label = tmp_label.astype(np.int)
        # print(tmp_label, label)
        # # img = (img * 255).astype(np.uint8)
        # img = draw_box(img, tmp_label)
        # cv2.imshow('im', img)
        # cv2.waitKey()

        if not self.l_is_normalized and self.normalize_label:
            if not im_path:
                raise Exception('The im_path not given when normalizing labels')
            img = cv2.imread(im_path)
            # print(im_path)
            clu_labels = np.zeros_like(label)
            img_shape = img.shape
            r = max(img_shape)
            clu_labels[:, (1, 3)] = label[:, (1, 3)] / r
            clu_labels[:, (2, 4)] = label[:, (2, 4)] / r

        clu_labels = None

        if self.clu_anchors:
            img = cv2.imread(im_path)
            # print(im_path)
            clu_labels = np.zeros_like(label)
            img_shape = img.shape
            r = max(img_shape)
            if self.l_is_normalized:
                clu_labels[:, (1, 3)] = label[:, (1, 3)] * (img_shape[1]/r)
                clu_labels[:, (2, 4)] = label[:, (2, 4)] * (img_shape[0]/r)
            else:
                clu_labels[:, (1, 3)] = label[:, (1, 3)] / r
                clu_labels[:, (2, 4)] = label[:, (2, 4)] / r
                if self.normalize_label:
                    label[:, (1, 3)] = label[:, (1, 3)] / img_shape[1]
                    label[:, (2, 4)] = label[:, (2, 4)] / img_shape[0]

            # h, w, _ = img.shape
            # tmp_label = np.zeros((label.shape[0], 4), dtype=np.float32)
            # tmp_label[:, (0, 2)] = label[:, (1, 3)] * w
            # tmp_label[:, (1, 3)] = label[:, (2, 4)] * h
            # tmp_label = xywh2xyxy(tmp_label[:, :4])
            # tmp_label = tmp_label.astype(np.int)
            # print(tmp_label, label)
            # # img = (img * 255).astype(np.uint8)
            # img = draw_box(img, tmp_label)
            # cv2.imshow('im', img)
            # cv2.waitKey()
        return label, im_path, clu_labels

    def cache_all_img(self, slice=None):
        pre_fix = "Caching all images to RAM"

        def _resize_im(params):
            im = cv2.imread(params[0])
            s = params[2]
            h, w, _ = im.shape
            r = min(h / s[1], w / s[0])
            im = cv2.resize(im, (0, 0), fx=r, fy=r)
            if not params[3]:
                labels = params[1]
                labels[:, 1:] = labels[:, 1:] * r
            else:
                labels = params[1]
            return params[0], im, labels

        if slice:
            data_path = self.paths[slice[0]: slice[1]]

            datas = [(d[1], self.datas[d[1]], self.img_size, self.l_is_normalized) for d in data_path]
            loader = ThreadPool(8).imap(_resize_im, datas)
            datas = {p[0]: (p[1], p[2]) for p in loader}
        else:
            loader = ThreadPool(8).imap(_resize_im, zip(*zip(*self.datas.items()),
                                                        repeat(self.img_size),
                                                        repeat(self.l_is_normalized)))
            loader = tqdm.tqdm(loader, desc=pre_fix, total=len(self.paths), ncols=100)
            datas = {p[0]: (p[1], p[2]) for p in loader}
            loader.close()
            print('Caching completed')
        self.datas = datas
        # print(self.datas)
        # return datas

    def load_img(self, im_label):
        img = cv2.imread(im_label[0])
        img_shape = img.shape
        if img_shape[0] != self.img_size[0] or img_shape[1] != self.img_size[1]:
            # img, label = dpl.resize_crop_im_label(img, label, self.img_size)
            img, label = resize_img(img, im_label[1], self.img_size)
        else:
            label = im_label[1]
        # img=img.astype(np.uint8)
        # print(img.dtype)
        return img, label, im_label[0]

    def read_img(self, im_path, label):
        img = cv2.imread(im_path)
        img_shape = img.shape
        if img_shape[0] != self.img_size[0] or img_shape[1] != self.img_size[1]:
            # img, label = dpl.resize_crop_im_label(img, label, self.img_size)
            img, label = resize_img(img, label, self.img_size)
        # img=img.astype(np.uint8)
        # print(img.dtype)
        return img, label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        tmp_path = self.paths[index]
        if not self.cache_imgs:
            # if self.cache2ram or self.read_l_from_cache:
            #     label = self.datas[tmp_path[1]]
            # else:
            #     label, _ = self.load_labels(tmp_path)
            im_labels = (tmp_path[1], self.datas[tmp_path[1]])
            img, labels, _ = self.load_img(im_labels)
        else:
            im_labels = self.datas[tmp_path[1]]
            img, labels, _ = self.load_img(im_labels)
        labels = tr.from_numpy(labels)
        img = tr.from_numpy(img)

        return img, labels

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        new_labels = []
        for i, label in enumerate(labels):
            l_shape = label.shape
            new_label = tr.zeros((l_shape[0], l_shape[1] + 1), dtype=tr.float32)
            new_label[:, 0] = i
            new_label[:, 1:] = label[:, :]
            new_labels.append(new_label)
        return tr.stack(imgs, dim=0), tr.cat(new_labels, dim=0)