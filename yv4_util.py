import numpy as np
import cv2
import os

from xml.dom import minidom
import Data_process as dpl


def draw_box(im, boxes, classes, col):
    def draw_single_box(box):
        box = box.astype(int)
        # print(im.shape, im.dtype, box[:4], col)
        xmin, ymin, xmax, ymax = box[:4]
        temp_im = cv2.rectangle(im, (xmin, ymin), (xmax, ymax), col, 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = classes[box[4] - 1]
        cv2.putText(im, text, (box[0], box[1]-5), font, 0.8, col, 1)
        return temp_im
    for box in boxes:
        im = draw_single_box(box)
    return im


def data_path(im_path, labels_path):
    if im_path[-1] != '\\':
        im_path += '\\'
    if labels_path[-1] != '\\':
        labels_path += '\\'
    label_names = os.listdir(labels_path)
    pic_paths = [im_path + name[:-3] + 'jpg' for name in label_names]
    label_paths = [labels_path + name for name in label_names]
    return pic_paths, label_paths


def read_anchor(path, scales, separator):
    fs = open(path, 'r')
    anchors_row = fs.readline().rstrip('\n')
    anchors_wh = anchors_row.split(separator)
    #anchors_wh.reverse()
    anchors_wh = [wh.split(',') for wh in anchors_wh]
    anchors_wh = [[float(w), float(h)] for w, h in anchors_wh]
    anchors_wh = np.asarray(anchors_wh)
    anchors_wh = np.reshape(anchors_wh, (scales, -1, 2))
    return anchors_wh


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []
    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def conf_nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    best_bboxes = []

    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])
        best_bbox = bboxes[max_ind]
        best_bboxes.append(best_bbox)
        bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
        weight = np.ones((len(iou),), dtype=np.float32)

        assert method in ['nms', 'soft-nms']

        if method == 'nms':
            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0

        if method == 'soft-nms':
            weight = np.exp(-(1.0 * iou ** 2 / sigma))

        bboxes[:, 4] = bboxes[:, 4] * weight
        score_mask = bboxes[:, 4] > 0.
        bboxes = bboxes[score_mask]

    return best_bboxes


def single_target_num(bboxes):
    best_bboxes = []
    max_ind = np.argmax(bboxes[:, 5])
    best_bbox = bboxes[max_ind]
    best_bboxes.append(best_bbox)

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def bbox_iou0(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area


def produce_labels(true_boxes, anchors, scales, output_size, max_bbox_per_scale, threshold=0.4):
    lens = true_boxes.shape[-1] + 1
    anchor_per_scale = anchors.shape[1]
    num_scales = anchors.shape[0]
    label = [np.zeros((output_size[i], output_size[i], anchor_per_scale,
                       lens)) for i in range(num_scales)]
    bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(num_scales)]
    bbox_count = np.zeros((num_scales,))

    for bbox in true_boxes:
        bbox_coor = bbox[:4]
        bbox_class_ind = np.argmax(bbox[4:])

        onehot = np.zeros(lens - 5, dtype=np.float)
        onehot[bbox_class_ind] = 1.0
        uniform_distribution = np.full(lens - 5, 1.0 / (lens - 5))
        deta = 0.01
        smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / scales[:, np.newaxis]

        iou = []
        exist_positive = False
        for i in range(num_scales):
            anchors_xywh = np.zeros((anchor_per_scale, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[i]

            iou_scale = bbox_iou0(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > threshold

            if np.any(iou_mask):
                xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                label[i][yind, xind, iou_mask, :] = 0
                label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                label[i][yind, xind, iou_mask, 4:5] = 1.0
                label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
                bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                bbox_count[i] += 1

                exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / anchor_per_scale)
            best_anchor = int(best_anchor_ind % anchor_per_scale)
            xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
            label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

            bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1
    label.extend(bboxes_xywh)
    return label


def nonecla_label_producer(true_boxes, anchors, scales, output_size, max_bbox_per_scale, threshold=0.4):
    lens = true_boxes.shape[-1] + 1
    anchor_per_scale = anchors.shape[1]
    num_scales = anchors.shape[0]
    label = [np.zeros((output_size[i], output_size[i], anchor_per_scale,
                       lens)) for i in range(num_scales)]
    bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(num_scales)]
    bbox_count = np.zeros((num_scales,))

    for bbox in true_boxes:
        bbox_coor = bbox[:4]

        bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / scales[:, np.newaxis]

        iou = []
        exist_positive = False
        for i in range(num_scales):
            anchors_xywh = np.zeros((anchor_per_scale, 4))
            anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
            anchors_xywh[:, 2:4] = anchors[i]

            iou_scale = bbox_iou0(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
            iou.append(iou_scale)
            iou_mask = iou_scale > threshold

            if np.any(iou_mask):
                xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                label[i][yind, xind, iou_mask, :] = 0
                label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                label[i][yind, xind, iou_mask, 4:5] = 1.0

                bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
                bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                bbox_count[i] += 1

                exist_positive = True

        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / anchor_per_scale)
            best_anchor = int(best_anchor_ind % anchor_per_scale)
            xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            label[best_detect][yind, xind, best_anchor, :] = 0
            label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
            label[best_detect][yind, xind, best_anchor, 4:5] = 1.0

            bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
            bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
            bbox_count[best_detect] += 1
    label.extend(bboxes_xywh)
    return label


def data_process(im, labels, input_size):
    im_shape = im.shape
    if im_shape[0] != input_size[0] or im_shape[1] != input_size[1]:
        im, labels = dpl.resize_crop_im_label(im, labels, input_size)
        #print('out',im.shape)
    return (im, labels)


def process_per_item(im_path, label_path, cla):
    class_node = 'name'
    bnd_nodes = ['xmin', 'ymin', 'xmax', 'ymax']
    im = cv2.imread(im_path)
    if type(im) != np.ndarray:
        print(im_path, type(im))
    label = dpl.readlables(label_path, cla, class_node, bnd_nodes)
    return im/255.0, label
