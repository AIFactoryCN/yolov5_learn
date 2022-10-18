import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import random
import logging
import platform


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str

def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 可复现
    # mark:
    # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法
    torch.backends.cudnn.deterministic = True
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    # 版权声明：本文为CSDN博主「AlanBupt」的原创文章
    # 原文链接：https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317
    torch.backends.cudnn.benchmark = False

    # 更快, 但是有损
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True

def set_logging(name=None, level=logging.INFO):
    # Sets level and returns logger
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)


LOGGER = logging.getLogger("yolov5")  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))  # emoji safe logging


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def use_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    g = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)
        if isinstance(v, bn):
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g[0].append(v.weight)
    
    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'optimizer {name} not implemented')
    
    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})

    return optimizer

def draw_bbox(image, left, top, right, bottom, confidence, classes, color=(0, 255, 0), thickness=1):
    
    left = int(left + 0.5)
    top = int(top + 0.5)
    right = int(right + 0.5)
    bottom = int(bottom + 0.5)
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
    
    if classes == -1:
        text = f"{confidence:.2f}"
    else:
        text = f"[{classes}]{confidence:.2f}"
    cv2.putText(image, text, (left + 3, top - 5), 0, 0.5, (0, 0, 255), 1, 16)


def draw_norm_bboxes(image, bboxes, color=(0, 255, 0), thickness=1):
    '''
    绘制归一化的边框
    参数：
        image[ndarray]:         图像
        bboxes[Nx4/Nx5/Nx6]:    框信息，列数可以是4、5、6，顺序是[cx, cy, width, height, confidence, classes]，基于图像大小进行归一化的框
    '''

    image_height, image_width = image.shape[:2]
    for obj in bboxes:
        cx, cy, width, height = obj[:4] * [image_width, image_height, image_width, image_height]
        left = cx - (width - 1) * 0.5
        top = cy - (height - 1) * 0.5
        right = cx + (width - 1) * 0.5
        bottom = cy + (height - 1) * 0.5

        confidence = 0
        if len(obj) > 4:
            confidence = obj[4]

        classes = -1
        if len(obj) > 5:
            classes = obj[5]

        draw_bbox(image, left, top, right, bottom, confidence, classes, color, thickness)


def draw_pixel_bboxes(image, bboxes, color=(0, 255, 0), thickness=1):
    '''
    绘制边框，基于left, top, right, bottom标注
    '''
    for obj in bboxes:
        left, top, right, bottom = [int(item) for item in obj[:4]]

        confidence = 0
        if len(obj) > 4:
            confidence = obj[4]

        classes = -1
        if len(obj) > 5:
            classes = obj[5]

        draw_bbox(image, left, top, right, bottom, confidence, classes, color, thickness)