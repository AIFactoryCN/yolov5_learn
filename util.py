import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import re
import random
import logging
import platform
import time
import torchvision
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import glob

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
    # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，
    # 其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
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


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
        return y
    
def xyxy2xywhn(x, w=640, h=640, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    '''
    将预测的坐标信息coords(相对img1_shape)转换回相对原图尺度(img0_shape)
    params:
        img1_shape: 缩放后的图像大小  [H, W]=[384, 640]
        coords: 预测的box信息[target_numbers, x1y1x2y2] 这个预测信息是相对缩放后的图像尺寸(img1_shape)的
        img0_shape: 原图的大小  [H, W, C]=[375, 670, 3]
        ratio_pad: 缩放过程中的缩放比例以及pad, 一般不传入
    return: 
        coords: 相对原图尺寸（img0_shape）的预测信息
    '''
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    '''
    c.clamp_(a, b): 将矩阵c中所有的元素约束在[a, b]中间
                    如果某个元素小于a, 就将这个元素变为a;
                    如果元素大于b,就将这个元素变为b
        这里将预测得到的xyxy做个约束，是因为当物体处于图片边缘的时候，预测值是有可能超过图片大小的
    param
        boxes: 输入时是缩放到原图的预测结果[target_numbers, x1y1x2y2]
                输出时是缩放到原图的预测结果，并对预测值进行了一定的约束，防止预测结果超出图像的尺寸
        shape: 原图的shape [H, W, C]=[375, 670, 3]
    '''
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45, 
                        classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    '''
    params:
        prediction: [batch, num_anchors_cell(3个预测层), (x+y+w+h+1+num_classes)] = [1, 25200, 85]  3个anchor的预测结果总和
        conf_threshold: 筛选，将分数过低的预测框删除
        classes: 是否nms后只保留特定的类别 默认为None
        agnostic: 确保多类别时只进行类内nms
        multi_label: 是否是多标签，一般是True
        max_det: 每张图片的最大目标个数，默认300
    '''

    num_classes = prediction.shape[2] - 5     # number of classes
    selected_mask = prediction[..., 4] > conf_threshold  # candidates 先使用置信度阈值过滤一次

    # Checks 检查传入的conf_thres和iou_thres两个阈值是否符合范围
    assert 0 <= conf_threshold <= 1, f'Invalid Confidence threshold {conf_threshold}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_threshold <= 1, f'Invalid IoU {iou_threshold}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels)预测物体宽度和高度的大小范围 
    max_nms = 30000  # 每个图像最多检测物体的个数
    time_limit = 10.0  # seconds to quit after
    multi_label &= num_classes > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    # 存取最终筛选结果的预测框信息
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x_new = x.clone()
        # 过滤小于和大于 框最大的宽高限制
        x_new[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # 取出满足置信度的框
        x_new = x_new[selected_mask[index]]  # confidence

        # 没有满足条件的框，结束这轮进行下一张
        if not x.shape[0]:
            continue

        # conf
        x_new = x.clone()
        x_new[:, 5:] = x[:, 5:] * x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (cx, cy, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x_new[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # 是否是多标签  nc>1  一般是True
        if multi_label:
            # 因为是多类别，根据obj_conf * cls_conf得到的阈值，对多类别框进行筛选
            i, j = (x[:, 5:] > conf_threshold).nonzero(as_tuple=False).T
            # pred = [10, xyxy+score+class] [10, 6]
            # unsqueeze(1): [10] -> [10, 1] add batch dimension
            # box[i]: [10, 4] xyxy
            # pred[i, j + 5].unsqueeze(1): [10, 1] score  对每个i取第（j+5）个位置的值（第j个class的值cls_conf）
            # j.float().unsqueeze(1): [10, 1] class
            x = torch.cat((box[i], x_new[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x_new[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        # 多类别时得到我们需要的类别nms结果
        if classes is not None:
            # 如果张量tensor中存在一个元素为True, 那么返回True; 只有所有元素都是False时才返回False
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 没有目标框就直接进行下一张
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # 如果框数量大于max_nms，就需要排序
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS 只做类内nms
        '''
            不同类别的框位置信息加上很大的数但又不同的数，
            使得做nms时不同类别的框就不会互相影响，是都类别做nms的小trick'''
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[index] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
        
    return output # num_matched x (5 + num_classes)


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, num_classes, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.num_classes, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            nc, nn = self.num_classes, len(names)  # number of classes, names
            sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
            labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True, vmin=0.0,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)   # 返回一个降序索引
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]   # 得到重新排序后对应的 tp, conf, pre_cls

    # Find unique classes 对类别去重, 因为计算ap是对每类进行
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    # 对每一个类别进行遍历处理
    for ci, c in enumerate(unique_classes):
        # i: 记录着所有预测框是否是c类别框   是c类对应位置为True, 否则为False
        i = pred_cls == c
        # n_l: gt框中的c类别框数量
        n_l = (target_cls == c).sum()  # number of labels
        # n_p: 预测框中c类别的框数量
        n_p = i.sum()  # number of predictions

        # 如果没有预测到 或者 ground truth没有标注 则略过类别c
        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            # tp[i] 可以根据i中的的True/False觉定是否删除这个数  所有tp中属于类c的预测框
            #       如: tp=[0,1,0,1] i=[True,False,False,True] b=tp[i]  => b=[0,1]
            # a.cumsum(0)  会按照对象进行累加操作
            # 一维按行累加如: a=[0,1,0,1]  b = a.cumsum(0) => b=[0,1,1,2]   而二维则按列累加
            # fpc: 类别为c 顺序按置信度排列 截至到每一个预测框的各个iou阈值下FP个数 最后一行表示c类在该iou阈值下所有FP数
            # tpc: 类别为c 顺序按置信度排列 截至到每一个预测框的各个iou阈值下TP个数 最后一行表示c类在该iou阈值下所有TP数
            fpc = (1 - tp[i]).cumsum(0)  # fp[i] = 1 - tp[i]
            tpc = tp[i].cumsum(0)

            # Recall
            # Recall=TP/(TP+FN)  加一个1e-16的目的是防止分母为0
            # n_l=TP+FN=num_gt: c类的gt个数=预测是c类而且预测正确+预测不是c类但是预测错误
            # recall: 类别为c 顺序按置信度排列 截至每一个预测框的各个iou阈值下的召回率
            recall = tpc / (n_l + 1e-16)  # recall curve
            # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的recall值  r=[nc, 1000]  每一行从小到大
            # 这里r的范围是[cls_nums, 1000]，这里是为了统一尺寸，利用插值限定了范围。每一列表示不同的iou阈值
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            # Precision=TP/(TP+FP)
            # precision: 类别为c 顺序按置信度排列 截至每一个预测框的各个iou阈值下的精确率
            precision = tpc / (tpc + fpc)  # precision curve
            # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的precision值  p=[nc, 1000]
            # 这里p的范围同样是[cls_nums, 1000]，这里是为了统一尺寸，利用插值限定了范围。每一列表示不同的iou阈值
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # 这里的召回率与准确率本质上是根据iou阈值为0.5来进行计算的，因为线性插值的时候使用的是recall[:, 0]和precision[:, 0]
            # 插值后的r:[nc, 1000], p:[nc, 1000]

            # AP from recall-precision curve
            # 对c类别, 分别计算每一个iou阈值(0.5~0.95 10个)下的mAP
            for j in range(tp.shape[1]):
                # 这里执行10次计算ci这个类别在所有mAP阈值下的平均mAP  ap[nc, 10], 依次循环计算不同阈值下的iou
                # 在当前类别下，根据每个阈值下的召回率与查全率来map（就算不规则图像的面积，也就是使用了一个定积分计算ap）
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    # 计算F1分数 P和R的调和平均值  综合评价指标
    # 我们希望的是P和R两个越大越好, 但是P和R常常是两个冲突的变量, 经常是P越大R越小, 或者R越大P越小 所以我们引入F1综合指标
    # 不同任务的重点不一样, 有些任务希望P越大越好, 有些任务希望R越大越好, 有些任务希望两者都大, 这时候就看F1这个综合指标了
    # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的f1值  f1=[nc, 1000]
    f1 = 2 * p * r / (p + r + 1e-16)

    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')

# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path