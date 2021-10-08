import torch as tr
import numpy as np
import torchvision as trv


def iou(b1, b2):
    eps = 1e-7
    top_rightx, top_righty = tr.min(b1[0], b2[:, 0]), tr.min(b1[1], b2[:, 1])
    bot_leftx, bot_lefty = tr.max(b1[2], b2[:, 2]), tr.max(b1[3], b2[:, 3])

    inter_rightx, inter_righty = tr.min(b1[2], b2[:, 2]), tr.min(b1[3], b2[:, 3])
    inter_leftx, inter_lefty = tr.max(b1[0], b2[:, 0]), tr.max(b1[1], b2[:, 1])

    u_a = (bot_leftx - top_rightx) * (bot_lefty - top_righty)
    i_a = (inter_rightx - inter_leftx).clamp(0) * (inter_righty - inter_lefty).clamp(0)

    return i_a/(u_a+eps)


def NMS(labels, cind=-1, conf_ind=-2, nc=None, thresh=0.2):
    c_list = []
    boxes = []
    if not nc:
        c = labels[:, 5:]
        nc = labels[:, 5:].shape[-1]
        c = c.argmax(dim=-1)[:, None]
        c_conf = labels[:, 5:][c]*labels[:, 4:5]
        labels = tr.cat((labels[:, :4], c_conf, c), dim=-1)
    for c in range(nc):
        mask = labels[:, cind] == c
        c_list.append(labels[mask])
    for box in c_list:
        count = 0
        while box.shape[0]:
            values, ind_c = box[:, conf_ind].max(0)
            true_box = box[ind_c]
            boxes.append(true_box)
            box = box[tr.arange(box.shape[0], device=box.device) != ind_c]
            count += 1
            ious = iou(true_box, box)
            mask = ious < thresh
            box = box[mask]
    return tr.stack(boxes)

