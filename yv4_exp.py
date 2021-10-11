import time
import torchvision as trv

import cv2
import torch as tr
import network as yv4
import pickle
import os
import glob
import tqdm
import numpy as np
import NMS
import Utility as ut


class detector:
    def __init__(self, model, check_p, img_size, device, show_image, original_image=True, cof_th=0.3):
        self.model = model
        self.show_image = show_image
        self.device = device
        self.anchors = tr.as_tensor(check_p['anchors'], dtype=tr.float32).to(device)
        self.cls = check_p['cls_names']
        self.img_size = img_size
        self.cof_th = cof_th
        self.original_image = original_image

    def img_prc(self, img, img_size):
        h, w, _ = img.shape
        hr, wr = img_size[0] / h, img_size[1] / w
        img = tr.from_numpy(img).to(device).permute(2, 0, 1) / 255.0
        if hr < wr:
            self.ra = 1/hr
            new_im = tr.zeros((3, *img_size), dtype=img.dtype, device=device)
            resize = trv.transforms.Resize([int(hr * h), int(hr * w)])
            img = resize(img)
            _, h, w = img.shape
            self.offset = (img_size[1] - w) // 2
            self.is_w = True
            new_im[:, :h, self.offset: w+self.offset] = img
        else:
            self.ra = 1/wr
            new_im = tr.zeros((3, *img_size), dtype=img.dtype, device=device)
            resize = trv.transforms.Resize([int(wr * h), int(wr * w)])
            img = resize(img)
            _, h, w = img.shape
            self.offset = (img_size[0] - h) // 2
            self.is_w = False
            # print(new_im.shape, img.shape)
            # print(off_set, h+off_set)
            new_im[:, self.offset: h+self.offset, :w] = img
        return new_im[None, :, :, :]

    def pre_detect(self, outputs):
        tmp_out = []
        for i, p in enumerate(outputs):
            out_shape = p.shape
            anchor = self.anchors[i] * tr.tensor(out_shape, dtype=tr.float32, device=self.device)[[3, 2]]
            p = p.view((out_shape[0], 3, len(self.cls) + 5, out_shape[2], out_shape[3])).permute(0, 1, 3, 4, 2)
            p = p.sigmoid()
            p[..., :2] = p[..., :2] * 2. - 0.5
            p[..., 2:4] = (p[..., 2:4] * 2) ** 2 * anchor[:, None, None, :]
            gridx = tr.arange(out_shape[3], device=self.device).repeat(out_shape[2], 1)[:, :, None]
            gridy = tr.arange(out_shape[2], device=self.device).repeat(out_shape[3], 1).T[:, :, None]
            grid = tr.cat((gridx, gridy), dim=-1)[None, None, :, :, :]
            p[..., :2] = p[..., :2] + grid
            mask = p[..., 4] > self.cof_th
            xscale = self.img_size[0] / out_shape[2]
            yscale = self.img_size[1] / out_shape[3]

            p = p[mask]
            # print(p.shape[0])
            if p.shape[0] == 0:
                continue
            else:
                p[:, (0, 2)] = p[:, (0, 2)] * xscale
                p[:, (1, 3)] = p[:, (1, 3)] * yscale
                cc, cind = p[..., 5:].max(dim=-1)
                c_conf = p[..., 5:].max(-1)[0] * p[..., 4]
                p = tr.cat((p[..., :4], c_conf[..., None], cind[..., None]), dim=-1)
                p = p.view((-1, 1 + 5))
                p[:, :4] = ut.xywh2xyxy(p[:, :4])
                tmp_out.append(p)
                # boxes = tr.cat(tmp_out, dim=0)
        if len(tmp_out) == 0:
            return None
        else:
            return tr.cat(tmp_out, dim=0)

    def detect(self, o_img, img_size, path=None):
        if self.show_image and self.original_image:
            self.img = o_img
        img = self.img_prc(o_img, img_size)
        outputs = model(img)
        outputs = self.pre_detect(outputs)
        if outputs is not None:
            boxes = NMS.NMS(outputs, nc=nc)
            if self.original_image:
                if not self.is_w:
                    boxes[:, (1, 3)] = boxes[:, (1, 3)] - self.offset
                    boxes[:, :4] = boxes[:, :4] * self.ra
                else:
                    boxes[:, (0, 2)] = boxes[:, (0, 2)] - self.offset
                    boxes[:, :4] = boxes[:, :4] * self.ra

                if self.show_image:
                    self.img = ut.draw_box(self.img, boxes, cls)
                    cv2.imshow('img', self.img)
                    cv2.waitKey(1)
                    if path:
                        cv2.imwrite(path, self.img)
            else:
                if self.show_image:
                    img = img.permute(1, 2, 0) * 255.0
                    img = img.uint8().detach().cpu().numpy()
                    img = ut.draw_box(img, boxes, cls)
                    cv2.imshow('img', img)
                    cv2.waitKey(1)
                if path:
                    cv2.imwrite(path, img)
            return boxes.detach().cpu().numpy()
        else:
            if self.original_image:
                cv2.imshow('img', self.img)
                cv2.waitKey(1)
                if path:
                    cv2.imwrite(path, self.img)
            else:
                img = img.permute(1, 2, 0) * 255.0
                img = img.uint8().detach().cpu().numpy()
                cv2.imshow('img', img)
                cv2.waitKey(1)
                if path:
                    cv2.imwrite(path, img)
            return None


def read_img_path(dir):
    img_format = ['jpg', 'png', 'tiff', 'bmp']
    paths = []
    for fm in img_format:
        glob_name = os.path.join(dir, '*.'+fm)
        paths.extend(glob.glob(glob_name))
    return paths


if __name__=="__main__":
    data_path = r'/media/jr/SOME/AutoSense/For_testing'
    video_path = r'F:\RDD\natural_v\fog_2.mp4'
    model_path = r'F:\RDD\AAR2.pt'
    save_dir = r'/home/jr/result1'

    cof_th = 0.25
    # nc = 3
    im_szs = [672, 672]
    data_type = 'video' #dir video cam

    device = tr.device('cuda:0')
    ckpt = tr.load(model_path)
    anchors = ckpt['anchors']
    cls = ckpt['cls_names']
    nc = len(cls)
    model_state = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}

    model = yv4.YOLOV4(len(cls), 3, 32, 3)

    model.load_state_dict(ckpt['model_state_dict'])
    model.cuda(device)
    model.eval()

    dt = detector(model, ckpt, im_szs, device, show_image=True, cof_th=cof_th)

    if data_type=='dir':
        data_paths = read_img_path(data_path)
        count = 0
        for b in tqdm.tqdm(data_paths, ncols=100):
            o_img = cv2.imread(b)
            path = os.path.join(save_dir, '{}.jpg'.format(count))
            dt.detect(o_img, im_szs, path)
            count += 1
    elif data_type=='video':
        cap = cv2.VideoCapture(video_path)
        succes, o_img = cap.read()
        while succes:
            dt.detect(o_img, im_szs,)
            succes, o_img = cap.read()
    else:
        cap = cv2.VideoCapture(0)
        succes, o_img = cap.read()
        while succes:
            dt.detect(o_img, im_szs,)
