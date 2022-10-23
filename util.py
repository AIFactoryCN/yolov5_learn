import cv2
import math
import time
import torch
import random
import logging
import platform
import numpy as np
import torchvision
import torch.nn as nn

from copy import deepcopy


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

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        # self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA

        self.ema = deepcopy(model.module if False else model).eval()  # FP32 EMA

        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            # msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            msd = model.module.state_dict() if False else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


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


def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45, 
                        classes=None, agnostic=False, multi_label=False, max_detections=300):
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
        max_detections: 每张图片的最大目标个数，默认300
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
        keep_index = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        if keep_index.shape[0] > max_detections:  # limit detections
            keep_index = keep_index[:max_detections]

        output[index] = x[keep_index]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
        
    return output # num_matched x (5 + num_classes)


