import numpy as np
import cv2
import torch as tr


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


def draw_box(img, boxes, cls_names=None, clor=(0, 0, 255)):
    img = (img*255).astype(np.uint8) if img.max() <= 1 else img
    def drawer(img, box):
        box = box.cpu().detach().numpy() if not isinstance(box, np.ndarray) else box
        cof = box[4]
        box = box.astype(np.int)
        xmin, ymin, xmax, ymax = box[:4]
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), clor, 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = cls_names[box[5]]+": {:.02}".format(cof**0.4)
        cv2.putText(img, text, (box[0], box[1]-5), font, 0.8, clor, 1)
        return img
    for box in boxes:
        img = drawer(img, box)
    return img


def normalizer(im, label):
    h, w, _ = im.shape
    label[:, (0, 2)] = label[:, (0, 2)]/w
    label[:, (1, 3)] = label[:, (1, 3)]/h
    return im, label


def im_padding(im, window_size):
    h, w, _ = im.shape
    sub_x = window_size[1] - w
    sub_y = window_size[0] - h
    top = 0
    left = 0
    if sub_x > 0 and sub_y > 0:
        left = sub_x // 2
        right = sub_x - left
        top = sub_y // 2
        bottom = sub_y - top
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
    elif sub_x > 0:
        left = sub_x // 2
        right = sub_x - left
        im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT)
    else:
        top = sub_y // 2
        bottom = sub_y - top
        im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
    return im, (left, top)


def resize_im_by2(im):
    h, w, _ = im.shape
    im = cv2.resize(im, (int(w*0.5), int(h*0.5)))
    return im


def crop_im(im, point, window_size):
    h, w, _ = im.shape
    hh = window_size[0]//2
    hw = window_size[1]//2
    point = np.minimum(point, (w - hh, h - hw))
    start_yp = max(0, int(point[1] + 0.5) - hh)
    start_xp = max(0, int(point[0] + 0.5) - hw)
    end_yp = start_yp + window_size[0]
    end_xp = start_xp + window_size[1]
    #print(start_yp, start_xp, end_yp, end_xp)
    return im[start_yp: end_yp, start_xp: end_xp], (start_xp, start_yp)


def chose_im_resize_method(im, output_size, previous_p, prev_wh, thresholds):
    previous_p = previous_p[0]
    temp_size = np.asarray(output_size)
    lens = len(thresholds)
    ith = 0
    scale = 1
    for i in range(lens):
        if prev_wh[0] > thresholds[i] or prev_wh[1] > thresholds[i]:
            ith = lens - i
            scale = pow(2, ith)
            scale = int(scale)
            break
    im, start_p = crop_im(im, previous_p, (output_size[0]*scale, output_size[1]*scale))
    for i in range(ith):
        im = resize_im_by2(im)
    h, w, _ = im.shape
    if h < output_size[0] or w < output_size[1]:
        im, residue = im_padding(im, output_size)
        return im, scale, start_p, residue
    else:
        return im, scale, start_p, None


def decode_xy(label, scale, start_p, residue):
    if residue is None:
        label[..., 0:4] = label[..., 0:4] * scale
        label[..., (0, 2)] = label[..., (0, 2)] + start_p[0]
        label[..., (1, 3)] = label[..., (1, 3)] + start_p[1]
    else:
        label[..., (0, 2)] = label[..., (0, 2)] - residue[0]
        label[..., (1, 3)] = label[..., (1, 3)] - residue[1]
        label[..., :4] = label[..., :4] * scale
        label[..., (0, 2)] = label[..., (0, 2)] + start_p[0]
        label[..., (1, 3)] = label[..., (1, 3)] + start_p[1]
    return label


def resize_crop_im(im, output_size, crop_num):
    trans_size = np.asarray(output_size)
    h, w, _ = im.shape
    input_size = np.asarray((h, w))
    sclaes = trans_size / input_size
    min_scale = np.amax(sclaes)
    im_list = []
    if input_size[0] > input_size[1]:
        im = cv2.resize(im, (int(w * min_scale), int(h * min_scale)))
        start_p = 0
        end_p = im.shape[0]
        for i in range(crop_num):
            if i % 2 == 1:
                if end_p - output_size[0] < 0:
                    temp_p = output_size[0]//2
                    im_list.append((im[temp_p: temp_p + output_size[0], :], temp_p))
                    return im_list
                else:
                    im_list.append((im[end_p - output_size[0]: end_p, :], end_p - output_size[0]))
                    end_p = end_p - output_size[0]
            else:
                if start_p + output_size[0] > im.shape[0]:
                    temp_p = im.shape[0] - output_size[0]//2
                    im_list.append((im[temp_p - output_size[0]: temp_p, :], temp_p - output_size[0]))
                else:
                    im_list.append((im[start_p: start_p + output_size[0], :], start_p))
                    start_p = start_p + output_size[0]
    else:
        im = cv2.resize(im, (int(w * min_scale), int(h * min_scale + 0.5)))
        start_p = 0
        end_p = im.shape[1]
        for i in range(crop_num):
            if i % 2 == 1:
                if end_p - output_size[1] < 0:
                    temp_p = output_size[1]//2
                    im_list.append((im[:, temp_p: temp_p + output_size[1]], temp_p))
                    return im_list
                else:
                    im_list.append((im[:, end_p - output_size[1]: end_p], end_p - output_size[1]))
                    end_p = end_p - output_size[1]
            else:
                if start_p + output_size[1] > im.shape[1]:
                    temp_p = im.shape[1] - output_size[1]//2
                    im_list.append((im[:, temp_p - output_size[1]: temp_p], temp_p - output_size[1]))
                else:
                    im_list.append((im[:, start_p: start_p + output_size[1]], start_p))
                    start_p = start_p + output_size[1]
    return im_list
