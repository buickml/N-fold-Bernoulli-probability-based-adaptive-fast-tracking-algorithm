import cv2
import torch as tr
import os
import glob
import time

import tqdm

import MY_TOOLS.Data_process_lib as dpl
from collections import Iterable
import random
import torch.multiprocessing as mp
from itertools import repeat
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
from multiprocessing.pool import ThreadPool
import cupy as cp


def resize_img(img, label, out_size):
    re_img = np.zeros((*out_size, img.shape[2]), dtype=img.dtype)
    # print(re_img.shape)
    h, w, _ = img.shape
    hr, wr, = out_size[0]/h, out_size[1]/w
    if hr <= wr:
        img = cv2.resize(img, (0, 0), fx=hr, fy=hr)
        h, w, _ = img.shape
        margin = out_size[1] - w
        offset = random.randint(0, margin)
        label[:, 1] = label[:, 1]*w/out_size[1] + (offset/out_size[1])
        label[:, 3] = label[:, 3]*w/out_size[1]
        re_img[:h, offset: offset+w, :] = img

    else:
        img = cv2.resize(img, (0, 0), fx=wr, fy=wr)
        h, w, _ = img.shape
        margin = out_size[0] - h
        offset = random.randint(0, margin)
        label[:, 2] = label[:, 2]*h/out_size[0] + (offset/out_size[0])
        label[:, 4] = label[:, 4]*h/out_size[0]
        re_img[offset: offset+h, :w, :] = img
    # print(label)
    return re_img, label


def draw_box(im, boxes, classes=None, col=(0, 0, 255)):
    def draw_single_box(box):
        box = box.astype(int)
        # print(im.shape, im.dtype, box[:4], col)
        xmin, ymin, xmax, ymax = box[:4]
        temp_im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), col, 3)
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
    y[:, (0, 1)] = x[:, (0, 1)] - (x[:, (2, 3)]/2)
    y[:, (2, 3)] = x[:, (0, 1)] + (x[:, (2, 3)]/2)
    return y


def data_enhance(img, labels, enhance_type, random_pram=None, pram=None):
    # bbs = BoundingBoxesOnImage([BoundingBox(*v) for v in label], img.shape)
    if enhance_type == 'flip':
        tmp = random.randint(0, 2)
        if tmp == 0:
            lr_img = np.fliplr(img)
            labels[:, 1] = 2*(0.5 - labels[:, 1]) + labels[:, 1]
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
        p = random.randint(0, lens-1)
        temp_im = img[:, :, :]
        gama_img = temp_im ** random_pram[p]
        return gama_img, labels

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
                print('Removed enhancement types:'+' '.join(rm_type))
        self.aug_key = list(self.aug_fn.keys())
        self.nf = nf if nf else len(self.aug_key)
        self.aug_first = aug_first
        self.mos = mosaic
        self.aug_seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # 水平翻转图像
            iaa.Flipud(0.5),  # 竖直反转
            # 高斯模糊
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),
            # affine
            iaa.Sometimes(0.5, iaa.Affine(translate_px=(-10, 10), rotate=(-30, 30), shear=(-10, 10))),
            # #增强或减弱图片的对比度
            # iaa.Sometimes(0.6, iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))),
            # #增强或减弱色温
            iaa.Sometimes(0.5, iaa.MultiplyHueAndSaturation((0.5, 1.1), per_channel=True)),
        ], random_order=True)

    def _augment(self, img, labels, enhance_type, random_pram=None, pram=None):
        # if enhance_type == 'flip':
        #     h, w, _ = img.shape
        #     tmp = random.randint(0, 3)
        #     if tmp == 0:
        #         lr_img = np.fliplr(img)
        #         labels[:, 0] = 2 * (w//2 - labels[:, 0]) + labels[:, 0]
        #         img, labels = lr_img, labels
        #     elif tmp == 1:
        #         ud_img = np.flipud(img)
        #         labels[:, 1] = 2 * (h//2 - labels[:, 1]) + labels[:, 1]
        #         img, labels = ud_img, labels
        #     elif tmp == 2:
        #         img = np.fliplr(img)
        #         labels[:, 0] = 2 * (w//2 - labels[:, 0]) + labels[:, 0]
        #
        #         lrup_img = np.flipud(img)
        #         labels[:, 1] = 2 * (h//2 - labels[:, 1]) + labels[:, 1]
        #         img, labels = lrup_img, labels
        #     else:
        #         img, labels = img, labels
        bbs = [BoundingBox(x1=l[0], y1=l[1], x2=l[2], y2=l[3]) for l in labels]

        bbs = BoundingBoxesOnImage(bbs, shape=img.shape)
        # Augment BBs and images.
        img, labels = self.aug_seq(image=img, bounding_boxes=bbs)
        labels = np.array([[l.x1, l.y1, l.x2, l.y2] for l in labels.bounding_boxes])

        if enhance_type == 'gama':
            p = random.sample(random_pram, 1)[0]
            if p==1.0:
                img, labels = img, labels
            else:
                temp_im = img[:, :, :] / 255.0
                img = ((temp_im ** p) * 255).astype(np.uint8)

        # tmp_label = np.zeros((labels.shape[0], 4), dtype=np.float32)
        # tmp_label[:, (0, 2)] = labels[:, (0, 2)]
        # tmp_label[:, (1, 3)] = labels[:, (1, 3)]
        # # tmp_label = xywh2xyxy(tmp_label[:, :4])
        # tmp_label = tmp_label.astype(np.int)
        # print(tmp_label, labels)
        # # re_img = (img * 255).astype(np.uint8)
        # re_img = draw_box(img, tmp_label)
        # cv2.imshow('im', img)
        # cv2.waitKey()
        # # labels[:, (1, 3)] = labels[:, (1, 3)]/img_szes[1]
        # # labels[:, (2, 4)] = labels[:, (2, 4)]/img_szes[0]

        return img, labels


    def augment(self, img, labels):
        aug = random.sample(self.aug_key, self.nf)
        h, w, _ = img.shape
        for a in aug:
            img, tmp = self._augment(img, labels[:, 1:], a, self.aug_fn[a])
            tmp[:, (0, 2)] = np.clip(tmp[:, (0, 2)], 0, w)
            tmp[:, (1, 3)] = np.clip(tmp[:, (1, 3)], 0, h)
            # print(tmp)
            labels[:, 1:] = tmp
        return img, labels


    def mosiac(self, datas, img_szes, lim_pix=20):

        def merg_labels(label, lxy, mxy, thr=0.5, xyxy=True): #xyxy
            if not xyxy:
                label[:, 1:] = xywh2xyxy(label[:, 1:])
            l_x1, l_x2 = label[:, 1] - mxy[0], label[:, 3] - mxy[0]
            r_x1, r_x2 = label[:, 1] - mxy[2], label[:, 3] - mxy[2]
            u_y1, u_y2 = label[:, 2] - mxy[1], label[:, 4] - mxy[1]
            b_y1, b_y2 = label[:, 2] - mxy[3], label[:, 4] - mxy[3]
            w = np.maximum(label[:, 3]-label[:, 1], 1e-6)
            h = np.maximum(label[:, 4]-label[:, 2], 1e-6)

            l_mask = (l_x1 < 0) & (l_x2 > 0)
            r_mask = (r_x1 < 0) & (r_x2 > 0)
            u_mask = (u_y1 < 0) & (u_y2 > 0)
            b_mask = (b_y1 < 0) & (b_y2 > 0)
            norm_mask = (l_x1 >= 0) & (r_x2 <= 0) & (u_y1 >= 0) & (b_y2 <= 0)

            # a_mask = (l_x1/w < -thr) | (u_y1/h < -thr) | (b_y2/h > thr) | (r_x2/w > thr)
            margin = np.minimum(l_x1/w, 0) + np.minimum(u_y1/h, 0) + np.minimum(-b_y2/h, 0) + np.minimum(-r_x2/w, 0)
            a_mask = margin < -thr

            mask = (l_mask | r_mask | u_mask | b_mask) & ~a_mask
            mask = mask | norm_mask
            label = label[mask]
            label[:, (1, 3)] = label[:, (1, 3)] - mxy[0]
            label[:, (2, 4)] = label[:, (2, 4)] - mxy[1]
            np.clip(label[:, 1], 0, mxy[2]-mxy[0], label[:, 1])
            np.clip(label[:, 3], 0, mxy[2]-mxy[0], label[:, 3])
            np.clip(label[:, 2], 0, mxy[3]-mxy[1], label[:, 2])
            np.clip(label[:, 4], 0, mxy[3]-mxy[1], label[:, 4])
            label[:, (1, 3)] = label[:, (1, 3)] + lxy[0]
            label[:, (2, 4)] = label[:, (2, 4)] + lxy[1]
            # label[:, 1:] = xyxy2xywh(label[:, 1:])
            return label

        im_ls = []
        for img, label in datas:
            img_shape = img.shape

            u_r = min(img_szes[1]/img_shape[1], img_szes[0]/img_shape[0])
            w_pix, h_pix = (label[:, -2:].T) * u_r
            if np.min(h_pix) >= lim_pix and np.min(w_pix) >= lim_pix:
                min_wr = np.min(lim_pix/w_pix)
                min_hr = np.min(lim_pix/h_pix)
                min_r = min(min_hr, min_wr)
                r = random.uniform(min_r, u_r)
                img = cv2.resize(img, (0, 0), fx=r, fy=r)
                label[:, 1:] = label[:, 1:] * r
                label[:, 1:] = xywh2xyxy(label[:, 1:])
                if self.aug_first:
                    img, label = self.augment(img, label)
                im_ls.append((img, label))
            else:
                r = u_r
                img = cv2.resize(img, (0, 0), fx=r, fy=r)
                label[:, 1:] = label[:, 1:] * r
                label[:, 1:] = xywh2xyxy(label[:, 1:])
                if self.aug_first:
                    img, label = self.augment(img, label)
                im_ls.append((img, label))

        d = random.randint(0, 255)
        re_img = np.full((*img_szes, 3), d)
        if len(datas)==4:
            xcor, ycor = random.randint(int(0.2*img_szes[1]), int(0.8*img_szes[1])), \
                         random.randint(int(0.2*img_szes[0]), int(0.8*img_szes[0]))
            label_list = []
            for i, (img, label) in enumerate(im_ls):
                img_shape = img.shape
                if i==0:
                    cw, ch = min(xcor, img_shape[1]), min(ycor, img_shape[0])
                    ax1, ay1 = xcor-cw, ycor-ch
                elif i==1:
                    cw, ch = min((img_szes[1]-xcor), img_shape[1]), min(ycor, img_shape[0])
                    ax1, ay1 = xcor, ycor - ch
                elif i==2:
                    cw, ch = min(xcor, img_shape[1]), min((img_szes[0]-ycor), img_shape[0])
                    ax1, ay1 = xcor-cw, ycor
                else:
                    cw, ch = min((img_szes[1]-xcor), img_shape[1]), min((img_szes[0]-ycor), img_shape[0])
                    ax1, ay1 = xcor, ycor
                ax2, ay2 = ax1 + cw, ay1 + ch
                bx1, by1 = random.randint(0, img_shape[1]-cw), random.randint(0, img_shape[0]-ch)
                bx2, by2 = bx1 + cw, by1 + ch

                # print(i)
                re_img[ay1: ay2, ax1: ax2] = img[by1: by2, bx1: bx2]
                # label[:, 1:] = xywh2xyxy(label[:, 1:])
                lxy = [ax1, ay1, ax2, ay2]
                mxy = [bx1, by1, bx2, by2]
                label_list.append(merg_labels(label, lxy, mxy))
            labels = np.concatenate(label_list, axis=0)
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

    def __call__(self, im_path_label, im_sze):
        if self.mos:
            if isinstance(im_path_label, list) and len(im_path_label)==4:
                datas = []
                for path, l in im_path_label:
                    if not isinstance(path, np.ndarray):
                        tmp_im = (cv2.imread(path))
                    else:
                        tmp_im = path
                    cv2.waitKey()
                    tmp_im = tmp_im
                    h, w, _ = tmp_im.shape
                    label = np.copy(l)
                    label[:, (1, 3)] = label[:, (1, 3)] * w
                    label[:, (2, 4)] = label[:, (2, 4)] * h
                    datas.append((tmp_im, label))
                img, labels = self.mosiac(datas, im_sze)
            else:
                raise Exception('The images path is not enough or type wrong when using mosaic augmentation')
        else:
            if not isinstance(im_path_label[0], np.ndarray):
                tmp_im = (cv2.imread(im_path_label[0]))
            else:
                tmp_im = im_path_label[0]
            tmp_im = tmp_im
            label = np.copy(im_path_label[1])
            h, w, _ = tmp_im.shape
            label[:, (1, 3)] = label[:, (1, 3)] * w
            label[:, (2, 4)] = label[:, (2, 4)] * h
            label[:, 1:] = xywh2xyxy(label[:, 1:])
            img, labels = self.augment(tmp_im, label)
            label[:, 1:] = xyxy2xywh(label[:, 1:])
        return img, labels


class dataset(tr.utils.data.Dataset):
    def __init__(self, path, img_size, label_info, cache_imgs=False, na=3, ns=3, prosc_fn=None, augment=None):
        self.cache_imgs = cache_imgs
        label_dir = path[0]
        img_dir = path[1]
        self.na = na
        self.ns = ns
        self.augment = augment
        if augment:
            if 'mosaic' in augment.keys():
                self.mosaic = augment['mosaic']
                del augment['mosaic']
            else:
                self.mosaic = False
            self.augment = Augment(True, augment, nf=2, mosaic=self.mosaic)
        self.label_info = label_info
        self.img_size = img_size
        self.check_label_info()
        if label_info['label type']=='xml':
            if 'class names' not in label_info or not isinstance(label_info['class names'], Iterable):
                raise Exception('You should provide "class names" when the "label type" is xml')
            glob_name = os.path.join(label_dir, '*.xml')
            label_paths = glob.glob(glob_name)
        elif label_info['label type']=='txt':
            glob_name = os.path.join(label_dir, '*.txt')
            label_paths = glob.glob(glob_name)
        else:
            glob_name = os.path.join(label_dir, '*.txt')
            label_paths = glob.glob(glob_name)
        img_format = ['jpg', 'png', 'tiff', 'bmp']

        self.paths = []
        for fm in img_format:
            self.paths.extend([(l, os.path.join(os.path.join(img_dir, os.path.basename(l)[:-4] + '.'+fm))) for l in label_paths
                          if os.path.exists(os.path.join(img_dir, os.path.basename(l)[:-4] + '.' + fm))])
        if self.read_l_from_cache:
            cache_path = os.path.join(os.path.dirname(label_dir), 'cache')
            # print(cache_path)
            if not os.path.exists(cache_path):
                print('The cache file is missing.')
                self.cache2local = True
                self.cach_label()
            else:
                self.datas = self.read_caching_label(cache_path)
                if len(self.datas) != len(self.paths):
                    self.paths = [(l, i) for l, i in self.paths if i in self.datas.keys()]
                    print("Waring: The cached data doesn't match the non-cache data. "
                          "The cached data lens is {0} and the non-cached data lens is {1}".format(len(self.datas), len(self.paths)))
        else:
            if self.cache2ram:
                self.cach_label()
        if cache_imgs:
            self.cache_all_img()
        self.prosc_fn = prosc_fn if prosc_fn else None


    def check_label_info(self):
        if not 'label type' in self.label_info.keys():
            print('Warning: The "label type" is not contained in label info dict, therefore, '
                  'the labels type will automatically set to "yolo type".')
            self.label_info['class first'] = True
            self.l_is_normalized = True
        else:
            if self.label_info['label type']=='xml':
                self.l_is_normalized = False
            elif self.label_info['label type']=='txt':
                if not 'label is normalized' in self.label_info.keys():
                    raise Exception('When the if "label is normalized" should given.')
                else:
                    self.l_is_normalized = self.label_info['label is normalized']
            else:
                self.l_is_normalized = True
        if not "class first" in self.label_info.keys():
            raise Exception('"class first" info should be contained in label info.')
        else:
            if 'transform to class first' not in self.label_info.keys():
                print('Waring: the label will transform to class first by '
                      'default unless you set "transform to class first" to "False".')
                self.trans2_class_first = True
            else:
                self.trans2_class_first = self.label_info['transform to class first']
        if not "xyxy" in self.label_info.keys():
            raise Exception('The coordinate type of the label (xyxy: bool type) dosent provided.')
        if 'cache label' not in self.label_info.keys() and \
            'read label from cache' not in self.label_info.keys():
            raise Exception('Either of the keys, "cache label" and "read label from cache", should be given in label info.')
        elif not 'cache label' in self.label_info.keys():
            self.read_l_from_cache = self.label_info['read label from cache']
            if self.read_l_from_cache:
                self.cache2local = False
            else:
                self.cache2local = True
                self.cache2ram = True
        elif 'read label from cache' not in self.label_info.keys():
            self.cache2local = self.label_info['cache label']
            self.cache2ram = True
            if self.cache2local:
                self.read_l_from_cache = False
        else:
            self.read_l_from_cache = True
            self.cache2ram = True
        if not 'cache to RAM' in self.label_info.keys():
            print('All the labels will be cache into RAM unless you provide the key '
                  '("cache to RAM") value in dict label_info.')
            self.cache2ram = True
        else:
            self.cache2ram = self.label_info['cache to RAM']
        if 'normalize label' not in self.label_info.keys():
            print('Warning: it is not indicated if the label should normalize by the key of label_info "normalize label".'
                  'Therefore, the label will be normalize by default')
            self.nor_l = True
        else:
            self.nor_l = self.label_info['normalize label']
        if 'cluster anchors' not in self.label_info.keys():
            self.clu_anchors = True
        else:
            self.clu_anchors = self.label_info['cluster anchors']
        if 'to xywh' not in self.label_info.keys():
            print('waring: the data format is "xywh" by default.')
            self.to_wh = True
        else:
            self.to_wh = self.label_info['to xywh']


    def label_cluster(self, labels):
        from sklearn.cluster import KMeans
        if labels.shape[1] != 2:
            raise Exception('The labels second dimension has to be 2, when clustering anchors.')
        # labels = labels*self.img_size
        km = KMeans(n_clusters=self.na*self.ns, init='k-means++', max_iter=1000, precompute_distances=True)
        km.fit(labels)
        cluter_centers = km.cluster_centers_
        cluter_centers = np.sort(cluter_centers, axis=0)
        # print(cluter_centers)
        return cluter_centers.reshape((self.ns, self.na, -1))


    def cach_label(self):
        # print(self.paths)
        with mp.Pool(8) as pool:
            pbar = tqdm.tqdm(pool.imap_unordered(self.load_labels, self.paths), desc='Caching Labels', total=len(self.paths), ncols=100)
            # pbar = pool.imap_unordered(self.load_labels, self.paths)
            self.datas = {p[1]: p[0] for p in pbar}

        if self.clu_anchors:
            labels = list(self.datas.values())
            labels = np.concatenate(labels, axis=0)
            if self.label_info['class first']:
                self.anchors = self.label_cluster(labels[:, 3:])
            else:
                self.anchors = self.label_cluster(labels[:, 2:4])
        if self.cache2local:
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
        return labels


    def load_labels(self, paths):
        path = paths[0]
        im_path = paths[1]
        if self.label_info['label type'] == 'xml':
            label = dpl.readlables(path, self.label_info['class names'], with_onehot=False)
            if self.to_wh:
                if self.label_info['xyxy'] and self.label_info['class first']:
                    label[:, 1:] = xyxy2xywh(label[:, 1:])
                elif self.label_info['xyxy'] and not self.label_info['class first']:
                    label[:, :-1] = xyxy2xywh(label[:, :-1])
            else:
                if not self.label_info['xyxy'] and self.label_info['class first']:
                    label[:, 1:] = xywh2xyxy(label[:, 1:])
                elif not self.label_info['xyxy'] and not self.label_info['class first']:
                    label[:, :-1] = xywh2xyxy(label[:, :-1])
        elif self.label_info['label type'] == 'yolo_type':
            label = dpl.readlables(path, len(self.label_info['class names']), self.label_info['label type'])
        else:
            label = dpl.readlables(path, len(self.label_info['class names']), self.label_info['label type'])
            if self.to_wh:
                if self.label_info['xyxy'] and self.label_info['class first']:
                    label[:, 1:] = xyxy2xywh(label[:, 1:])
                elif self.label_info['xyxy'] and not self.label_info['class first']:
                    label[:, :-1] = xyxy2xywh(label[:, :-1])
            else:
                if not self.label_info['xyxy'] and self.label_info['class first']:
                    label[:, 1:] = xywh2xyxy(label[:, 1:])
                elif not self.label_info['xyxy'] and not self.label_info['class first']:
                    label[:, :-1] = xywh2xyxy(label[:, :-1])

        label = label.astype(np.float32)
        if self.label_info['class first']:
            if np.max(label[:, 1:]) <= 1:
                self.l_is_normalized = True
            else:
                self.l_is_normalized = False
        else:
            if np.max(label[:, :-1]) <= 1:
                self.l_is_normalized = True
            else:
                self.l_is_normalized = False

        if not self.label_info['class first'] and self.trans2_class_first:
            new_label = np.zeros_like(label)
            new_label[:, 0] = label[:, -1]
            new_label[:, 1:] = label[:, :-1]
            label = new_label
            # print(label)
        elif self.label_info['class first'] and not self.trans2_class_first:
            new_label = np.zeros_like(label)
            new_label[:, -1] = label[:, 0]
            new_label[:, :-1] = label[:, 1:]
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

        if not self.l_is_normalized and self.nor_l:
            if not im_path:
                raise Exception('The im_path not given when normalizing labels')
            img = cv2.imread(im_path)
            # print(im_path)
            img_shape = img.shape
            if self.trans2_class_first:
                label[:, (1, 3)] = label[:, (1, 3)] / img_shape[1]
                label[:, (2, 4)] = label[:, (2, 4)] / img_shape[0]
            else:
                label[:, (0, 2)] = label[:, (0, 2)] / img_shape[1]
                label[:, (1, 3)] = label[:, (1, 3)] / img_shape[0]

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
        return label, im_path


    def cache_all_img(self, slice=None):
        pre_fix = "Caching all images to RAM"

        def _resize_im(params):
            im = cv2.imread(params[0])
            s = params[2]
            h, w, _ = im.shape
            r = min(h/s[1], w/s[0])
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
            loader = ThreadPool(8).imap_unordered(_resize_im, datas)
            datas = {p[0]: (p[1], p[2]) for p in loader}
            loader.close()
        else:
            loader = ThreadPool(8).imap_unordered(_resize_im, zip(*zip(*self.datas.items()),
                                                repeat(self.img_size), repeat(self.l_is_normalized)))
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
        random.seed(time.time()+1)
        tmp_path = self.paths[index]
        if index == 0:
            random.shuffle(self.paths)
            # random.seed()
        if not self.cache_imgs:
            # if self.cache2ram or self.read_l_from_cache:
            #     label = self.datas[tmp_path[1]]
            # else:
            #     label, _ = self.load_labels(tmp_path)
            if self.mosaic:
                paths = random.sample(self.paths, 3)
                paths.append(tmp_path)
                im_labels = [(p, self.datas[p]) for _, p in paths]
            else:
                im_labels = (tmp_path[1], self.datas[tmp_path[1]])
                im_labels = self.load_img(im_labels)
        else:
            if self.mosaic:
                paths = random.sample(self.paths, 3)
                paths.append(tmp_path)
                im_labels = [self.datas[p] for _, p in paths]
            else:
                im_labels = self.datas[tmp_path[1]]
                im_labels = self.load_img(im_labels)

        img, labels = self.augment(im_labels, self.img_size)
        img = img / 255.0
        h, w, _ = img.shape
        labels[:, (1, 3)] = labels[:, (1, 3)]/w
        labels[:, (2, 4)] = labels[:, (2, 4)]/h


        # h, w, _ = img.shape
        # tmp_label = np.zeros((labels.shape[0], 4), dtype=np.float32)
        # tmp_label[:, (0, 2)] = labels[:, (1, 3)] * w
        # tmp_label[:, (1, 3)] = labels[:, (2, 4)] * h
        # tmp_label = xywh2xyxy(tmp_label[:, :4])
        # tmp_label = tmp_label.astype(np.int)
        # # print(tmp_label, labels)
        # img = (img*255).astype(np.uint8)
        # img = draw_box(img, tmp_label)
        # print(img.shape)
        # cv2.imshow('im', img)
        # cv2.waitKey()
            
        img = tr.from_numpy(img.astype(np.float32))
        label = tr.from_numpy(labels.astype(np.float32))

        return img.permute(2, 0, 1), label

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        new_labels = []
        for i, label in enumerate(labels):
            l_shape = label.shape
            new_label = tr.zeros((l_shape[0], l_shape[1]+1), dtype=label.dtype)
            new_label[:, 0] = i
            new_label[:, 1:] = label
            new_labels.append(new_label)
        return tr.stack(imgs, dim=0), tr.cat(new_labels, dim=0)