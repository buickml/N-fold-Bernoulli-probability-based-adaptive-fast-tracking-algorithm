# Loss functions

import torch as tr
import torch.nn as nn
import math
import cv2
import numpy as np


def draw_box(im, boxes, classes, col):
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


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (tr.min(b1_x2, b2_x2) - tr.max(b1_x1, b2_x1)).clamp(0) * \
            (tr.min(b1_y2, b2_y2) - tr.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = tr.max(b1_x2, b2_x2) - tr.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = tr.max(b1_y2, b2_y2) - tr.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytr/blob/master/utils/box/box_utils.py#L47
                # print(w2, h2, w1, h1)
                v = (4 / math.pi ** 2) * tr.pow(tr.atan(w2 / h2) - tr.atan(w1 / h1), 2)
                with tr.no_grad():

                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = tr.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - tr.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = tr.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = tr.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = tr.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = tr.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, anchors):
        super(ComputeLoss, self).__init__()
        self.anchors = anchors
        device = anchors.device
        # device = next(model.parameters()).device  # get model device
        # # h = model.hyp  # hyperparameters
        #
        # # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=tr.tensor([1.0], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=tr.tensor([1.0], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)  # positive, negative BCE targets

        # Focal loss
        g = 0.0  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        #
        # det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(anchors.shape[0], [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        # self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, 0
        self.BCEcls, self.BCEobj = nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss()
        # for k in 'na', 'nc', 'nl', 'anchors':
        #     setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, images):  # predictions, targets, model
        # print(p[0].shape)
        p = [r.view((r.shape[0], self.anchors.shape[0], r.shape[1] // self.anchors.shape[0],
                     r.shape[2], r.shape[3])).permute(0, 1, 3, 4, 2) for r in p]
        device = targets.device
        lcls, lbox, lobj = tr.zeros(1, device=device), tr.zeros(1, device=device), tr.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # print(pi.shape)
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = tr.zeros_like(pi[..., 0], device=device)  # target obj
            # print(b.shape)

            n = b.shape[0]  # number of targets

            if n:

                # print(tbox[i].shape, gi.shape, gj.shape)
                # org_box = tr.cat(
                #     (tbox[i][..., 0:1] + gi[:, None], tbox[i][..., 1:2] + gj[:, None], tbox[i][..., 2:]), dim=-1)
                # # print(org_box)
                # # org_box[..., 0:4] = org_box[..., 0:4] * stride
                # # org_box = tr.cat((org_box, box[..., 4:]), dim=-1)
                # # org_box[..., 0:4] = org_box[..., 0:4] * stride
                # org_box = tr.cat(
                #     (org_box[..., (0, 1)] - org_box[..., (2, 3)] / 2, org_box[..., (0, 1)] + org_box[..., (2, 3)] / 2),
                #     dim=-1)
                # for i in range(images.shape[0]):
                #     print(n)
                #     mask = i == b
                #     boxes = org_box[mask]
                #     boxes = boxes.cpu().numpy()
                #     img = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
                #     # print(pi.shape)
                #     h = pi.shape[2]
                #     rt = img.shape[1] / h
                #     print(img.shape, boxes.shape)
                #     img = draw_box(img, boxes * rt, ['Fighter'], (0, 0, 255))
                #     cv2.imshow('im', img)
                #     cv2.waitKey()

                ps = pi[b, a, gj, gi]  # prediction subset corresponding    to targets
                # print(ps.shape)
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = tr.cat((pxy, pwh), 1)  # predicted
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                # tobj[b, a, gj, gi] = 1.0 - tr.pow(iou.detach().clamp(0), 5).type(tobj.dtype)
                tobj[b, a, gj, gi] = 1.0 * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # if self.nc > 1:  # cls loss (only if multiple classes)
                t = tr.full_like(ps[:, 5:], self.cn, device=device)  # targets
                t[range(n), tcls[i]] = self.cp
                lcls += self.BCEcls(ps[:, 5:], t)  # BCE
                # print(lcls)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in tr.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
        #     if self.autobalance:
        #         self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        #
        # if self.autobalance:
        #     self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= 1.0  # self.hyp['box']
        lobj *= 1.0  # self.hyp['obj']
        lcls *= 1.0  # self.hyp['cls']
        bs = tobj.shape[0]  # batch size
        # print(lbox, lobj, lcls)

        loss = lbox + lobj + lcls
        return loss * bs, tr.cat((lbox, lobj, lcls, loss), dim=-1)

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        nl = len(p)
        na, nt = 3, targets.shape[0]  # number of anchors, targets
        all_na = nl * na
        tcls, tbox, indices, anch = [], [], [], []
        gain = tr.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = tr.arange(all_na, device=targets.device).float().view(1, all_na).repeat(nt, 1)  # same as .repeat_interleave(nt)
        targets = tr.cat((targets[:, None, :].repeat(1, all_na, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # biastmp
        off = tr.tensor([[0, 0],
                         [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                         ], device=targets.device).float() * g  # offsets

        # gt_ind = []
        # ctas = []
        # for i in range(3):
        #     gain[2:6] = tr.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        #     anchors = self.anchors[i] * gain[2:4]
        #     # Match targets to anchors
        #     t = targets * gain
        #     if nt:
        #         # Matches
        #         r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
        #         # print(r)
        #         cta = tr.max(r, 1. / r).max(2)[0]
        #         ctas.append(cta)
        #         j = cta < 5  # compare
        #         gt_ind.append(j)
        #
        # all_ind = tr.cat(gt_ind, dim=-1)
        # all_ind = [all_ind[ind*nt: (ind+1)*nt] for ind in range(na) if ind < na - 1]
        # all_ind = tr.cat(all_ind, dim=-1)
        # all_ind = tr.sum(all_ind, dim=[-1])
        # all_ind = all_ind == 0

        anchors = self.anchors.clone()

        for i in range(nl):
            gain[2:6] = tr.tensor(p[i].shape)[[3, 2, 3, 2]]
            targets[:, i*na: (i + 1)*na] = targets[:, i*na: (i + 1)*na] * gain
            anchors[i] = anchors[i] * gain[2:4]
        # print(anchors)
        anchors = anchors.view((-1, 2))
        if nt:
            r = targets[:, :, 4:6] / anchors[None, :, :]
            # print(r.shape, '--'*8)
            cta = tr.max(r, 1. / r).max(2)[0]
            # print(cta.shape, '&&'*8)
            ind = cta < 4
            c_j = ind.any(-1)
            if not c_j.all():
                non_ind = ~c_j
                ct = cta[non_ind]
                min_ind = ct.min(-1)[1]
                ct = tr.zeros_like(ct, dtype=tr.bool, device=targets.device)
                for i in range(ct.shape[0]):
                    ct[i, min_ind[i]] = True
                ind[non_ind] = ct
            # print(tr.sum(ind, dim=1), '**' * 8)
            ind = ind.T
            targets = targets.permute(1, 0, 2)

            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

        for i in range(nl):
            # Match targets to anchors
            gain[2:6] = tr.tensor(p[i].shape)[[3, 2, 3, 2]]
            t = targets[i * na: (i + 1) * na]
            t[..., 6] = t[..., 6] - (i * na)
            anchor = anchors[i * na: (i + 1) * na]
            if nt:
                j = ind[i * na: (i + 1) * na]

                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                # print(t)

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = tr.stack((tr.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (tr.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(tr.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchor[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


if __name__ == '__main__':
    tar = tr.arange(5) + 1
    targets = tar.view(5, 1).repeat(1, 7)
    nt = targets.shape[0]
    ai = tr.arange(9, device=targets.device).float().view(1, 9).repeat(nt, 1)  # same as .repeat_interleave(nt)
    targets = tr.cat((targets[:, None, :].repeat(1, 9, 1), ai[:, :, None]), 2)
    anchors = tr.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    if nt:
        r = targets[:, :, 4:6] / anchors[None, :, :]
        cta = tr.max(r, 1. / r).max(2)[0] * 4
        cta = tr.randint(0, 5, (5, 9))
        ind = cta < 1
        print(ind.shape)
        c_j = ind.any(-1)
        tmp = targets[ind]
        print(targets.shape)
        if not c_j.all():
            print('=' * 5, ~c_j)
            non_ind = ~c_j
            ct = cta[non_ind]
            print(ct.shape)
            min_ind = ct.min(-1)[1]
            # print(min_ind, ct)
            ct = tr.zeros_like(ct, dtype=tr.bool)
            for i in range(ct.shape[0]):
                ct[i, min_ind[i]] = True
            ind[non_ind] = ct
        print(ind)
        ind = ind.T
        targets = targets.permute(1, 0, 2)
        print(targets)
