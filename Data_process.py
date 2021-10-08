import sys
sys.path.append(r'E:\JarhinbekR\Python')
#import tensorflow as tf
import numpy as np
import cv2
import os
import multiprocessing as mp
import random
from xml.dom import minidom
import MY_TOOLS.commen_tools as ct
import torch as tr


def one_hot(cla, classes):
    lens = len(classes)
    oh = [0.0] * lens
    ind = classes.index(cla)
    oh[ind] = 1.0
    return oh


def readlables(path, clas, class_node = None, bnd_nodes = None):
    root = minidom.parse(path).documentElement
    names = root.getElementsByTagName(class_node)
    lens = len(names)
    boxes = []
    classes = []
    for i in range(lens):
        boxes.append([])
    for cla in root.getElementsByTagName(class_node):
            oh = one_hot(cla.childNodes[0].nodeValue, clas)
            classes.append(oh)
    for cor in bnd_nodes:
        temp = root.getElementsByTagName(cor)
        for i, c in enumerate(temp):
            boxes[i].append(int(c.childNodes[0].nodeValue))

    labels = [b.extend(c) for b, c in zip(boxes, classes)]
    return np.asarray(boxes)


'''
def Int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


def Bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


def TFRecord(images, lables, path_or_writer = None):
    if type(path_or_writer) == str:
        writer=tf.io.TFRecordWriter(path_or_writer)
    else:
        writer = path_or_writer
    num_examples = images.shape[0]
    squeezed_im = tf.reshape(images, [num_examples, -1])
    def creat_examp(image, lable, pixels):
        image_to_string = image.tostring()
        feature = {
            'pixels': Int64_feature(pixels),
            'label': Int64_feature(lable),
            'image': Bytes_feature(image_to_string)
        }
        features = tf.train.Features(feature = feature)
        return tf.train.Example(features = features)
    pixels = squeezed_im.shape[1]
    sess = tf.Session()
    images = images.eval(session = sess)
    examples = [creat_examp(im, la, pixels) for im, la in zip(images, lables)]
    list(map(lambda x: writer.write(x.SerializeToString()), examples))


def ReadTF(file_path, shape, cast = True, classes = None, one_hot = True):
    if type(file_path) == list:
        file_name_q = tf.train.string_input_producer(file_path)
    else:
        print(file_path)
        file_name_q = tf.train.string_input_producer([file_path])
    reader = tf.TFRecordReader()
    _, examples = reader.read(file_name_q)
    features = tf.io.parse_single_example(
        serialized=examples,
        features={
        'image': tf.io.FixedLenFeature([], tf.string),
        'pixels': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)
    })
    images = tf.io.decode_raw(features['image'], tf.uint8)
    labels = tf.cast(features['label'], tf.int32)
    pixels = tf.cast(features['pixels'], tf.int32)
    images = tf.reshape(images, shape)
    if cast:
        images = tf.cast(images, tf.float32)
    if one_hot:
        if not classes:
            raise('You must give noumber of classes')
        else:
            labels = tf.one_hot(labels, 10)
    return images, labels, pixels


def normolization(im, labels):
    h, w, _ = im.shape
    labels[..., (0, 2)] = labels[..., (0, 2)]/w
    labels[..., (1, 3)] = labels[..., (1, 3)]/w
    im = im/255.0
    return im, labels


def bath_out(file_path, shape, batch_size, capacity, num_threads = None):
    images, labels, _ = ReadTF(file_path, shape)
    return tf.train.batch([images, labels], batch_size, num_threads, capacity)


def get_file_path(path, filter_fun = None):
    if path[-1] != '/':
        path = path + '/'
    data_path = os.listdir(path)
    if filter_fun:
        names = filter(filter_fun, data_path)
        return [path + x for x in names]
    else:
        return [path + x for x in data_path]\
'''


def im_lable_padding(im, labels, output_size):
    h, w, _ = im.shape
    if h < output_size[1] and w < output_size[0]:
        final_im = np.zeros((*output_size, 3))
        reminder_x = output_size[0] - w
        reminder_y = output_size[1] - h
        start_xp = random.randint(0, reminder_x)
        start_yp = random.randint(0, reminder_y)
        final_im[start_yp: start_yp + h, start_xp: start_xp + w] = im[:,:,:]
        labels[..., (1, 3)] = labels[..., (1, 3)] + start_yp
        labels[..., (0, 2)] = labels[..., (0, 2)] + start_xp

        return final_im, labels
    elif h < output_size[0]:
        final_im = np.zeros((output_size[0], w, 3))
        reminder_y = output_size[0] - h
        start_yp = random.randint(0, reminder_y)
        final_im[start_yp: start_yp + h, :, :] = im[:, :, :]
        labels[..., (1, 3)] = labels[..., (1, 3)] + start_yp
        if w == output_size[0]:
            return final_im, labels
        else:
            return resize_padding_im_label(final_im, labels, output_size)
    else:
        final_im = np.zeros((h, output_size[1], 3))
        reminder_x = output_size[1] - w
        start_xp = random.randint(1, reminder_x)
        final_im[:, start_xp: start_xp + w, :] = im[:, :, :]
        labels[..., (0, 2)] = labels[..., (0, 2)] + start_xp
        if h == output_size[1]:
            return final_im, labels
        else:
            return resize_padding_im_label(final_im, labels, output_size)


def resize_padding_im_label(im, label, output_size):
    trans_size = np.asarray(output_size)
    temp_im = np.zeros((*output_size, 3))
    h, w, _ = im.shape
    input_size = np.asarray((h, w))
    sclaes = trans_size/input_size
    min_scale = np.amin(sclaes)
    classes = label[..., 4:]
    label = label[..., :4] * min_scale
    label = label.astype(int)
    if input_size[0] > input_size[1]:
        new_w = int(w * min_scale)
        start_p = random.randint(0, (input_size[1] - new_w) // 2)
        im = cv2.resize(im, (new_w, int(h * min_scale)))
        temp_im[:, start_p: start_p + new_w] = im[:, :, :]
        label[..., (0, 2)] = label[..., (0, 2)] + start_p
    else:
        new_h = int(h * min_scale)
        start_p = random.randint(0, (output_size[1] - new_h) // 2)
        im = cv2.resize(im, (int(w * min_scale), new_h))
        temp_im[start_p: start_p + new_h, :] = im[:, :, :]
        label[..., (1, 3)] = label[..., (1, 3)] + start_p
    return temp_im, np.concatenate([label, classes], axis=-1)


def crop_im_label(im, labels, output_size):
    h, w, _ = im.shape
    classes = labels[..., 4:]
    labels = labels[..., :4]
    if h > output_size[0] and w > output_size[1]:
        min_x = int(np.amin(labels[..., 0]))
        max_x = int(np.amax(labels[..., 2]))
        min_y = int(np.amin(labels[..., 1]))
        max_y = int(np.amax(labels[..., 3]))
        span_x = max_x - min_x
        span_y = max_y - min_y
        reminder_x = output_size[1] - span_x
        reminder_y = output_size[0] - span_y
        front_x = min((reminder_x, min_x))
        front_y = min((reminder_y, min_y))
        start_p = min_x - random.randint(0, front_x)
        start_p = min((start_p, w - output_size[1]))
        labels[..., (0, 2)] = labels[..., (0, 2)] - start_p
        temp_im = im[:, start_p: output_size[1] + start_p]
        start_p = min_y - random.randint(0, front_y)
        start_p = min((start_p, h - output_size[0]))
        labels[..., (1, 3)] = labels[..., (1, 3)] - start_p
        temp_im = temp_im[start_p: output_size[0] + start_p, :]
    elif h > output_size[0]:
        min_y = np.amin(labels[..., 1])
        max_y = np.amax(labels[..., 3])
        span_y = max_y - min_y
        reminder_y = output_size[0] - span_y
        front_y = min((reminder_y, min_y))
        start_p = min_y - random.randint(0, front_y)
        start_p = min((start_p, h - output_size[0]))
        labels[..., (1, 3)] = labels[..., (1, 3)] - start_p
        temp_im = im[start_p: output_size[0] + start_p, :]
        if w != output_size[1]:
            temp_im, labels = im_lable_padding(temp_im, np.concatenate([labels,
                                                classes], axis=-1), output_size)
    else:
        min_x = np.amin(labels[..., 0])
        max_x = np.amax(labels[..., 2])
        span_x = max_x - min_x
        reminder_x = output_size[1] - span_x
        front_x = min((reminder_x, min_x))
        start_p = min_x - random.randint(0, front_x)
        start_p = min((start_p, w - output_size[1]))
        labels[..., (0, 2)] = labels[..., (0, 2)] - start_p
        temp_im = im[:, start_p: output_size[1] + start_p]
        if h != output_size[0]:
            temp_im, labels = im_lable_padding(temp_im, np.concatenate([labels,
                                                classes], axis=-1), output_size)
    return temp_im, np.concatenate([labels, classes], axis=-1)


def resize_crop_im_label(row_im, row_label, output_size):
    trans_size = np.asarray(output_size)
    im = np.copy(row_im)
    h, w, _ = im.shape
    if h < output_size[0] or w < output_size[1]:
        return im_lable_padding(im, row_label, output_size)
    input_size = np.asarray((h, w))
    sclaes = trans_size/input_size
    min_scale = np.amax(sclaes)
    label = np.copy(row_label)
    classes = label[..., 4:]
    label = label[..., :4] * min_scale
    label = label.astype(int)
    if input_size[0] > input_size[1]:
        im = cv2.resize(im, (output_size[1], int(h * min_scale)))
        max_x = np.amax(label[..., 3], axis=-1)
        min_x = np.amin(label[..., 1], axis=-1)
        span = max_x - min_x
        if span > output_size[1]:
            return resize_padding_im_label(row_im, row_label, output_size)
        elif span + 10 < output_size[0]:
            reminder = output_size[0] - span
            front = min(random.randint(5, reminder - 5), min_x)
            cro_front = min_x - front
            cro_front = min(cro_front, int(w * min_scale) - 416)
            temp_im = im[cro_front: cro_front + output_size[0], :]
            label[..., (1, 3)] = label[..., (1, 3)] - cro_front
            #print('one', temp_im.shape)
        else:
            reminder = output_size[0] - span
            front = reminder//2
            temp_im = im[:, front: front+output_size[0]]
            label[..., (0, 2)] = front + (label[..., (0, 2)] - min_x)
            #print('two', temp_im.shape)
        #print('h big {}'.format(temp_im.shape))
    else:
        im = cv2.resize(im, (int(w * min_scale), output_size[0]))
        max_y = np.amax(label[..., 2], axis=-1)
        min_y = np.amin(label[..., 0], axis=-1)
        span = max_y - min_y
        if span > output_size[1]:
            return resize_padding_im_label(row_im, row_label, output_size)
        elif span + 10 < output_size[1]:
            reminder = output_size[1] - span
            front = min(random.randint(5, reminder - 5), min_y)
            cro_front = min_y - front
            cro_front = min(cro_front, int(w * min_scale) - output_size[1])
            temp_im = im[:, cro_front: cro_front + output_size[0]]
            #print('three', temp_im.shape)
            label[..., (0, 2)] = label[..., (0, 2)] - cro_front
        else:
            reminder = output_size[1] - span
            front = reminder // 2
            cro_front = min_y - front
            cro_front = min(cro_front, int(w * min_scale) - 416)
            temp_im = im[:, cro_front: cro_front + output_size[1]]
            #print('four', temp_im.shape)
            label[..., (0, 2)] = label[..., (0, 2)] - cro_front
        #print('w big {}'.format(int(h * min_scale)))
    return temp_im, np.concatenate([label, classes], axis=-1)


def resize_padding_im(im, output_size):
    trans_size = np.asarray(output_size)
    h, w, _ = im.shape
    temp_im = np.zeros((*output_size, 3))
    input_size = np.asarray((h, w))
    sclaes = trans_size / input_size
    min_scale = np.amin(sclaes)
    if input_size[0] > input_size[1]:
        new_w = int(w * min_scale)
        half_cape = (input_size[1] - new_w)//2
        im = cv2.resize(im, (new_w, int(h * min_scale)))
        temp_im[:, half_cape: half_cape + new_w] = im[:, :, :]
    else:
        new_h = int(h * min_scale)
        half_cape = (output_size[1] - new_h) // 2
        im = cv2.resize(im, (int(w * min_scale), new_h))
        temp_im[half_cape: half_cape + new_h, :] = im[:, :, :]
    return temp_im, half_cape


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


def crop_im_f_b(im, output_size):
    trans_size = np.asarray(output_size)
    h, w, _ = im.shape
    input_size = np.asarray((h, w))
    sclaes = trans_size / input_size
    min_scale = np.amax(sclaes)
    if input_size[0] > input_size[1]:
        im = cv2.resize(im, (int(w * min_scale), int(h * min_scale)))
        temp_h, temp_w, _ = im.shape
        im0 = im[:416, :]
        im1 = im[temp_h - 416:temp_h, :]
    else:
        im = cv2.resize(im, (int(w * min_scale), int(h * min_scale)))
        temp_h, temp_w, _ = im.shape
        im0 = im[:, :416]
        im1 = im[:, temp_w - 416:temp_w]
    return im0, im1


def crop_im_around_p(im, window_size, p):
    h, w, _ = im.shape
    p = p[0].astype(int)
    if h > window_size[0] and w > window_size[1]:
        scale = max(window_size[0]/h, window_size[1]/w)
        im = cv2.resize(im, (int(w*scale + 0.5), int(h*scale + 0.5)))
        h, w, _ = im.shape
        start_y = p[1] - window_size[0]//2
        start_y = max(0, start_y)
        start_y = min(start_y, h - window_size[0])
        start_x = p[0] - window_size[1]//2
        start_x = max(0, start_x)
        start_x = min(start_x, w - window_size[1])
        end_y = start_y + window_size[0]
        end_x = start_x + window_size[1]
        im = im[start_y: end_y, start_x: end_x]
        residue = np.asarray((start_x, start_y))
    elif h > window_size[0]:
        start_y = p[1] - window_size[0]//2
        start_y = max(0, start_y)
        start_y = min(start_y, h - window_size[0])
        im = im[start_y: end_y, :, :]
        residue = np.asarray((0, start_y))
    elif w > window_size[1]:
        start_y = 0
        start_x = p[0] - window_size[1]//2
        start_x = max(0, start_x)
        start_x = min(start_x, w - window_size[1])
        end_y = window_size[0]
        end_x = start_x + window_size[1]
        im = im[: , start_x: end_x]
        residue = np.asarray((start_x, 0))
    if h < window_size[0] and w < window_size[1]:
        ih, iw, _ = im.shape
        temp_im = np.zeros((*window_size, 3))
        start_y = (window_size[0] - h)//2
        start_x = (window_size[1] - w)//2
        temp_im[start_y: start_y+ih, start_x: start_x+iw, :] = im[:, :, :]
        im = temp_im
        residue = np.asarray(start_x, start_y)
    elif h < window_size[0]:
        ih, _, _ = im.shape
        start_y = (window_size[0] - h)//2
        temp_im = np.zeros((*window_size, 3))
        temp_im[start_y: start_y+ih, :, :] = im[:, :, :]
        im = temp_im
        residue = np.asarray(0, start_y)
    elif w < window_size[1]:
        _, iw, _ = im.shape
        temp_im = np.zeros((*window_size, 3))
        start_x = (window_size[1] - w)//2
        temp_im[:, start_x: start_x+iw, :] = im[:, :, :]
        im = temp_im
        residue = np.asarray(start_x, 0)
    return im, residue


def label_process(pro_fun, *param):
    return pro_fun(*param)


def read_from_spacial_reader(dir_path, read_fun, process_fun=None, bylist=False, filter_fun =None):
    file_paths = get_file_path(dir_path, filter_fun)
    datas = [read_fun(path) for path in file_paths]
    if process_fun:
        if bylist:
            return [process_fun(data) for data in datas]
        else:
            return process_fun(datas)
    else:
        return datas


def images_filter(type_list = ('.jpg', '.png', '.bmp')):
    def filter_fun(na):
        return na[-3:] in type_list
    return filter_fun


def enhancement(im, labels, filp = [1, 2, 3], gama = None):
    shape = im.shape
    h, w = shape[0], shape[1]
    centro_x, centro_y = w//2, h//2
    data = []
    y = []
    if 1 in filp:
        lr_labels = np.copy(labels)
        data.append(np.fliplr(im))
        mask_coor = labels[..., (0, 2)]
        mask = labels[..., (0, 2)] > centro_x
        mask_coor[mask] = w - mask_coor[mask]

        mask = labels[..., (0, 2)] < centro_x
        mask_coor[mask] = (centro_x - mask_coor[mask]) + centro_x
        lr_labels[..., (2, 0)] = mask_coor
        lr_labels.astype(int)
        y.append(lr_labels)
    if 2 in filp:
        ud_labels = np.copy(labels)
        data.append(np.flipud(im))
        mask_coor = labels[..., (1, 3)]
        mask = labels[..., (1, 3)] > centro_y
        mask_coor[mask] = h - mask_coor[mask]

        mask = labels[..., (1, 3)] < centro_y
        mask_coor[mask] = (centro_y - mask_coor[mask]) + centro_y
        ud_labels[..., (3, 1)] = mask_coor
        ud_labels.astype(int)
        y.append(ud_labels)
    if 3 in filp:
        cor_labels = np.copy(labels)
        temp_im = np.fliplr(im)
        mask_coor = labels[..., (0, 2)]
        data.append(np.flipud(temp_im))
        mask = labels[..., (0, 2)] > centro_x
        mask_coor[mask] = w - mask_coor[mask]

        mask = labels[..., (0, 2)] < centro_x
        mask_coor[mask] = (centro_x - mask_coor[mask]) + centro_x
        cor_labels[..., (2, 0)] = mask_coor

        mask_coor = labels[..., (1, 3)]
        mask = labels[..., (1, 3)] > centro_y
        mask_coor[mask] = h - mask_coor[mask]

        mask = labels[..., (1, 3)] < centro_y
        mask_coor[mask] = (centro_y - mask_coor[mask]) + centro_y
        cor_labels[..., (3, 1)] = mask_coor
        cor_labels.astype(int)
        y.append(cor_labels)
    return data, y


def chose_resieze_method(im, labels, output_size, thresh = 42):
    temp_label = (labels[..., 2:4] - labels[..., 0:2]) < thresh
    if np.any(temp_label):
        return crop_im_label(im, labels, output_size)
    else:
        return resize_crop_im_label(im, labels, output_size)


def trans_data(im, labels, window):
    window = np.asarray(window)
    wh = labels[:, 2: 4] - labels[:, 0: 2]
    center = (labels[:, 0: 2] + labels[:, 2: 4])//2
    reminder = window - wh
    start_y = int(max(0, center[:, 1] - (random.randint(0, reminder[0, 1]) + wh[:, 1]//2)))
    start_x = int(max(0, center[:, 0] - (random.randint(0, reminder[0, 0]) + wh[:, 0]//2)))
    end_y = start_y + window[0]
    end_x = start_x + window[1]
    temp_im = im[start_y: end_y, start_x: end_x, :]
    trans_label = center - np.asarray((start_x, start_y))
    temp_label = np.concatenate([trans_label - wh * 0.5, trans_label + wh * 0.5, labels[:, 4:]], axis=-1)
    return temp_im, temp_label


class tracker_data_producer(mp.Process):
    def __init__(self, queue, data_dirs, get_datafun, get_data_param, data_processor, extra_param,
                window_size, windows_num, batch_size, echo, margin=50, shuffle=True, num_process=1, buffer = 2):
        mp.Process.__init__(self)
        self.windows_num = windows_num
        self.window_size = window_size
        self.margin = margin
        self.get_data = get_datafun
        self.batch_size = batch_size
        self.mini_batch_size = batch_size//(windows_num + 1)
        self.reminder = batch_size%windows_num
        self.buffer = buffer
        self.windows_num = windows_num
        self.queue = queue
        self.get_data_param = get_data_param
        self.data_processor = data_processor
        self.data_paths = list(zip(*self.data_path(*data_dirs)))
        self.shuffle = shuffle
        self.extra_param = extra_param
        self.data_size = len(self.data_paths)
        if shuffle:
            random.shuffle(self.data_paths)
        self.mini_echos = self.data_size//self.mini_batch_size - 1
        self.mini_e_counter = 0
        self.echo = echo
        self.num_process=num_process

    def padding(self, im, labels):
        h, w, _ = im.shape
        sub_x = self.window_size[1] - w
        sub_y = self.window_size[0] - h
        if sub_x > 0 and sub_y > 0:
            left = sub_x // 2
            right = sub_x - left
            top = sub_y // 2
            bottom = sub_y - top
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)
            labels[..., (0, 2)] = labels[..., (0, 2)] + left
            labels[..., (1, 3)] = labels[..., (1, 3)] + top
        elif sub_x > 0:
            left = sub_x // 2
            right = sub_x - left
            im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT)
            labels[..., (0, 2)] = labels[..., (0, 2)] + left
        else:
            top = sub_y // 2
            bottom = sub_y - top
            im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT)
            labels[..., (1, 3)] = labels[..., (1, 3)] + top
        #im = fuf.draw_box(im, labels, ['Drogue'], (0, 0, 255))
        #cv2.imshow('img', im)
        #cv2.waitKey(1000)
        return im, labels

    def resize_im_by2(self, im, labels):
        h, w, _ = im.shape
        im = cv2.resize(im, (int(w*0.5), int(h*0.5)))
        labels[..., 0:4] = labels[..., 0:4] // 2
        #im = fuf.draw_box(im, labels, ['Drogue'], (0, 0, 255))
        #cv2.imshow('img', im)
        #cv2.waitKey(1000)
        return im, labels

    def data_path(self, im_path, labels_path):
        if im_path[-1] != '/':
            im_path += '/'
        if labels_path[-1] != '/':
            labels_path += '/'
        label_names = os.listdir(labels_path)
        pic_paths = [im_path + name[:-3] + 'jpg' for name in label_names]
        label_paths = [labels_path + name for name in label_names]
        return pic_paths, label_paths

    def trans_data(self, im, labels):
        h, w, _ = im.shape
        if labels.shape[0] != 1:
            labels = labels[0:1]
        window_size = np.asarray(self.window_size)
        wh = labels[:, 2: 4] - labels[:, 0: 2]
        center = (labels[:, 0: 2] + labels[:, 2: 4])//2
        reminder = window_size - wh
        start_y = int(max(0, center[:, 1] - (random.randint(0, reminder[0, 1]) + wh[:, 1]//2)))
        start_x = int(max(0, center[:, 0] - (random.randint(0, reminder[0, 0]) + wh[:, 0]//2)))
        start_y = min(start_y, h - window_size[0])
        start_x = min(start_x, w - window_size[1])
        end_y = start_y + window_size[0]
        end_x = start_x + window_size[1]
        temp_im = im[start_y: end_y, start_x: end_x, :]
        trans_label = center - np.asarray((start_x, start_y))
        trans_label = np.concatenate([trans_label - wh*0.5, trans_label + wh*0.5, labels[..., 4:]], axis=-1)
        return temp_im, trans_label

    def data_filter(self, pic_path, label_path):
        im, label = self.get_data(pic_path, label_path, *self.get_data_param)
        while (label[..., 2:4] - label[..., 0:2] > np.asarray(self.window_size) - self.margin).any():
            im, label = self.resize_im_by2(im, label)
        else:
            h, w, _ = im.shape
            if h < self.window_size[0] or w < self.window_size[1]:
                im, label = self.padding(im, label)
            return im, label

    def producer(self):
        counter = 0
        repeat = True
        end_p = 0
        while repeat:
            if self.queue.qsize() > self.buffer - 1:
                continue
            if self.mini_e_counter > self.mini_echos - 2:
                self.mini_e_counter = 0
                counter += 1
                end_p = 0
                if self.shuffle:
                    random.shuffle(self.data_paths)
            if counter == self.echo - 1:
                repeat = False
            #if counter == 0:
            start_p = end_p
            end_p += self.mini_batch_size
            end_p = min(end_p, self.data_size - 1)
            batch_data_paths = self.data_paths[start_p: end_p]
            datas = []
            for data_path in batch_data_paths:
                data = self.data_filter(*data_path)
                windows = [self.trans_data(*data) for i in range(self.windows_num)]
                datas.extend(windows)
            residue = self.batch_size - len(datas)
            while residue != 0 and end_p < self.data_size -1:
                data_path = self.data_paths[end_p]
                data = self.data_filter(*data_path)
                windows = [self.trans_data(*data) for i in range(min((residue, self.windows_num)))]
                datas.extend(windows)
                residue = self.batch_size - len(datas)
            self.mini_e_counter += 1
            if len(datas) != self.batch_size:
                self.mini_e_counter = self.mini_echos
                continue
            datas = [self.data_processor(im, label, *self.extra_param) for im, label in datas]
            datas = [(x[0], *x[2:]) for x in datas]
            self.queue.put(list(zip(*datas)))

    def run(self):
        try:
            p = []
            for i in range(self.num_process):
                p1 = mp.Process(target=self.producer, args=())
                p1.start()
                p.append(p1)
                random.shuffle(self.data_paths)
            for sub_p in p:
                sub_p.join()
        except (KeyboardInterrupt, SystemError, SystemExit):
            for sub_p in p:
                sub_p.terminate()
            print('Data processing process has been stoped')
            raise
        finally:
            for sub_p in p:
                sub_p.terminate()
            print('Data processing process has been stoped')
            raise


class muti_process_v_producer(mp.Process):
    def __init__(self, q, read_dir, fps, buff):
        mp.Process.__init__(self)
        self.q = q
        self.paths = self.data_path(read_dir)
        self.fps = fps
        self.buff = buff

    def data_path(self, path):
        if path[-1] != '/':
            path += '/'
        v_names = os.listdir(path)
        v_paths = [path + name for name in v_names]
        return v_paths

    def run(self):
        for path in self.paths:
            cap = cv2.VideoCapture(path)
            success = True
            count = 0
            while success:
                if self.q.qsize() == self:
                    continue
                success, frame = cap.read()
                if success:
                    if count % self.fps == 0:
                        self.q.put(frame)
                    count += 1
            cap.release()


class muti_process_batch(tr.multiprocessing.Process):
    def __init__(self, queue, im_dir, label_dir, get_datafun, process_fun, batch_size, method=(1, 2, 3),
                gamas = (), follow=True, repeat=True, shuffle=True,  buff_num=1, extra_param=None, num_process=2):
        tr.multiprocessing.Process.__init__(self)
        self.num_process = num_process
        self.get_datafun = get_datafun
        self.extra_parm = extra_param
        self.process_fun = process_fun
        self.repeat = repeat
        self.shuffle = shuffle
        im_paths, label_paths = self.data_path(im_dir, label_dir)
        self.datapaths = list(zip(im_paths, label_paths))
        self.echo = 0
        self.data_size = len(self.datapaths)
        self.enhance_method = method
        self.gamas = gamas
        self.batch_size = batch_size
        enhance = len(method) + len(gamas) + 1
        if enhance != 1:
            self.min_batch_size = batch_size//enhance
            self.remainder = batch_size % enhance
            self.all_echo = self.data_size//(batch_size//enhance) - (enhance - 1)
            self.enhance = enhance
        else:
            self.enhance = 0
            self.batch_size = batch_size
            self.all_echo = self.data_size//batch_size
        if shuffle:
            random.shuffle(self.datapaths)
        self.q = queue
        self.follow = follow
        self.buff_num = buff_num

    def data_path(self, im_path, labels_path):
        if im_path[-1] != '/':
            im_path += '/'
        if labels_path[-1] != '/':
            labels_path += '/'
        label_names = os.listdir(labels_path)
        pic_paths = [im_path + name[:-3] + 'jpg' for name in label_names]
        label_paths = [labels_path + name for name in label_names]
        return pic_paths, label_paths

    def enhancement(self, im, labels, filp=(1, 2, 3, 4), gamas=()):
        shape = im.shape
        h, w = shape[0], shape[1]
        centro_x, centro_y = w // 2, h // 2
        data = []
        y = []
        if 1 in filp:
            lr_labels = np.copy(labels)
            data.append(np.fliplr(im))
            mask_coor = labels[..., (0, 2)]
            mask = labels[..., (0, 2)] > centro_x
            mask_coor[mask] = w - mask_coor[mask]

            mask = labels[..., (0, 2)] < centro_x
            mask_coor[mask] = (centro_x - mask_coor[mask]) + centro_x
            lr_labels[..., (2, 0)] = mask_coor
            lr_labels.astype(int)
            y.append(lr_labels)
        if 2 in filp:
            ud_labels = np.copy(labels)
            data.append(np.flipud(im))
            mask_coor = labels[..., (1, 3)]
            mask = labels[..., (1, 3)] > centro_y
            mask_coor[mask] = h - mask_coor[mask]

            mask = labels[..., (1, 3)] < centro_y
            mask_coor[mask] = (centro_y - mask_coor[mask]) + centro_y
            ud_labels[..., (3, 1)] = mask_coor
            ud_labels.astype(int)
            y.append(ud_labels)
        if 3 in filp:
            cor_labels = np.copy(labels)
            temp_im = np.fliplr(im)
            mask_coor = labels[..., (0, 2)]
            data.append(np.flipud(temp_im))
            mask = labels[..., (0, 2)] > centro_x
            mask_coor[mask] = w - mask_coor[mask]

            mask = labels[..., (0, 2)] < centro_x
            mask_coor[mask] = (centro_x - mask_coor[mask]) + centro_x
            cor_labels[..., (2, 0)] = mask_coor

            mask_coor = labels[..., (1, 3)]
            mask = labels[..., (1, 3)] > centro_y
            mask_coor[mask] = h - mask_coor[mask]

            mask = labels[..., (1, 3)] < centro_y
            mask_coor[mask] = (centro_y - mask_coor[mask]) + centro_y
            cor_labels[..., (3, 1)] = mask_coor
            cor_labels.astype(int)
            y.append(cor_labels)
        if 4 in filp:
            clabels = np.copy(labels)
            y.append(clabels)
            data.append(1 - im)
        lens = len(gamas)
        if lens != 0:
            for gama in gamas:
                temp_im = im[:,:,:]
                temp_im = temp_im**gama
                data.append(temp_im)
                y.append(np.copy(labels))
        return data, y

    def enrich_enhance_q(self, queue, batch_paths, supply):
        if supply:
            im_labels = [self.get_datafun(im_path, label_path) for im_path, label_path in batch_paths]
            temp_im_labels = [self.process_fun(data, label, *self.extra_parm) for data, label in im_labels]
            im_labels = [(x[0], x[1]) for x in temp_im_labels]
            datas = [(x[0], x[1]) for x in temp_im_labels]
            for im, label in im_labels:
                enhanced_data = self.enhancement(im, label, self.enhance_method, self.gamas)
                for en_im, en_label in zip(*enhanced_data):
                    queue.put((en_im, en_label))
            im_labels = []
            if self.remainder:
                for j in range(self.remainder):
                    im_labels.append(queue.get())
        else:
            im_labels = [self.get_datafun(im_path, label_path) for im_path, label_path in batch_paths]
            temp_im_labels = [self.process_fun(data, label, *self.extra_parm) for data, label in im_labels]
            im_labels = [(x[0], x[1]) for x in temp_im_labels]
            datas = [(x[0], x[1]) for x in temp_im_labels]
            #print('31out:{} data_size:{} end_p:{}, start_p:{}'.format(len(im_labels), self.data_size, end_p, start_p))
            for im, label in im_labels:
                enhanced_data = self.enhancement(im, label, self.enhance_method, self.gamas)
                for en_im, en_label in zip(*enhanced_data):
                    queue.put((en_im, en_label))
            #print('32out{}'.format(len(im_labels)))
            im_labels = []
            for j in range(self.min_batch_size * (self.enhance - 1)):
                im_labels.append(queue.get())
            #print('33out{}'.format(len(im_labels)))
            if self.remainder:
                for j in range(self.remainder):
                    im_labels.append(queue.get())
        return im_labels, datas

    def producer(self):
        if self.enhance != 0:
            enchanced = mp.Queue()
            f_end_p = 0
            while True:
                datas = []
                if self.q.qsize() > self.buff_num - 1:
                    continue
                if self.echo > self.all_echo:
                    self.echo = 0
                    f_end_p = 0
                    if self.shuffle:
                        random.shuffle(self.datapaths)
                if self.echo == 0:
                    #print('1in')
                    start_p = f_end_p
                    f_end_p += self.min_batch_size * self.enhance
                    batch_paths = self.datapaths[start_p: f_end_p]
                    im_labels, datas = self.enrich_enhance_q(enchanced, batch_paths, True)
                    self.echo += 1
                    #print('1out')
                elif self.echo == self.all_echo:
                    #print('=='*20)
                    if enchanced.qsize() < self.batch_size:
                        self.echo += 1
                        continue
                    else:
                        im_labels = [enchanced.get() for j in range(self.batch_size)]
                else:
                    #print('3in')
                    start_p = f_end_p
                    if enchanced.qsize() < self.remainder:
                        supply = True
                        f_end_p += self.min_batch_size * self.enhance
                        f_end_p = min(f_end_p, self.data_size)
                        if (f_end_p - start_p)*self.enhance < (self.batch_size):
                            self.echo = self.all_echo
                            continue
                    else:
                        f_end_p += self.min_batch_size
                        if f_end_p > self.data_size:
                            self.echo = self.all_echo
                            continue
                        supply = False
                    batch_paths = self.datapaths[start_p: f_end_p]
                    im_labels, datas = self.enrich_enhance_q(enchanced, batch_paths, supply)
                    self.echo += 1
                    #print('34out{}'.format(len(im_labels)))
                temp = [self.process_fun(data, label, *self.extra_parm) for data, label in im_labels]
                datas.extend([(x[0], x[1]) for x in temp])
                self.q.put(list(zip(*datas)))

        else:
            while True:
                if self.q.qsize() == self.buff_num:
                    continue
                if self.echo == self.all_echo:
                    self.echo = 0
                    if self.shuffle:
                        random.shuffle(self.datapaths)
                start_p = self.echo * self.batch_size
                end_p = (self.echo + 1) * self.batch_size
                batch_paths = self.datapaths[start_p: end_p]
                im_labels = [self.get_datafun(im_path, label_path) for im_path, label_path in batch_paths]
                im_labels = [self.process_fun(data, label, *self.extra_parm) for data, label in im_labels]
                gt = list(zip(*im_labels))
                trgt_im = tr.tensor(gt[0], dtype=tr.float32, device='cuda:0')

                # print(gt[1])
                trgt_label = tr.tensor(np.array(gt[1]), device='cuda:0')
                self.q.put((trgt_im, trgt_label))
                # print(trgt_label.shape)
                self.echo += 1

    def run(self):
        try:
            self.child_p = []
            p = []
            datas = []
            p = tr.multiprocessing.spawn(fn=self.producer, nprocs=self.num_process)
            # for i in range(self.num_process):
            #     p1 = tr.multiprocessing.Process(target=self.producer, args=())
            #     # p1.daemon = True
            #     p1.start()
            #     self.child_p.append(p1.pid)
            #     p.append(p1)
            #     random.shuffle(self.datapaths)
            # for sub_p in p:
            #     sub_p.join()
            p.join()

        finally:
            for sub_p in p:
                sub_p.terminate()
                self.child_p = self.child_p[1:]
            print('Data processing process has been stoped')
            raise