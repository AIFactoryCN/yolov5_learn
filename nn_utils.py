import cv2
import torch
import torch.nn as nn
import math
import random
import numpy as np

from scipy.cluster.vq import kmeans
from tqdm import tqdm
from copy import deepcopy

class BBox:
    def __init__(self, x, y, r, b, landmark):
        
        self.x = x
        self.y = y
        self.r = r
        self.b = b
        self.landmark = landmark

    def __repr__(self):
        landmark_info = "HasLandmark" if self.landmark else "NoLandmark"
        return f"{{Face {self.x}, {self.y}, {self.r}, {self.b}, {landmark_info} }}"
    
    @property
    def left_top_i(self):
        return int(self.x), int(self.y)
    
    @property
    def right_bottom_i(self):
        return int(self.r), int(self.b)
    
    @property
    def center_i(self):
        return int((self.x + self.r) * 0.5), int((self.y + self.b) * 0.5)
    
    @property
    def center(self):
        return (self.x + self.r) * 0.5, (self.y + self.b) * 0.5
    
    @property
    def width(self):
        return self.r - self.x + 1
    
    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def location(self):
        return self.x, self.y, self.r, self.b

    @property
    def landmark_union(self):
        union = ()
        for point in self.landmark:
            union = union + tuple(point)
        return union
        
class ImageObject:
    def __init__(self, file):
        self.file = file
        self.bboxes = []

    def add(self, annotation):
        x, y, w, h = annotation[:4]
        r = x + w - 1
        b = y + h - 1
        landmark = None
        
        if len(annotation) == 20:
            # x, y, w, h, xyz, xyz, xyz, xyz, xyz, unknow
            landmark = []
            for i in range(5):
                px = annotation[i * 3 + 0 + 4]
                py = annotation[i * 3 + 1 + 4]
                pz = annotation[i * 3 + 2 + 4]
                
                if pz == -1:
                    landmark = None
                    break
                    
                landmark.append([px, py])
        self.bboxes.append(BBox(x, y, r, b, landmark))
        

def load_widerface_annotation(ann_file):
    with open(ann_file, "r") as f:
        lines = f.readlines()

    imageObject = None
    file = None
    images = []
    for line in lines:
        line = line.replace("\n", "")

        if line[0] == "#":
            file = line[2:]
            imageObject = ImageObject(file)
            images.append(imageObject)
        else:
            imageObject.add([float(item) for item in line.split(" ")])
    return images


def draw_gauss_np(heatmap, x, y, box_size):

    if not isinstance(box_size, tuple):
        box_size = (box_size, box_size)

    box_width, box_height = box_size
    diameter = min(box_width, box_height)

    height, width = heatmap.shape[:2]
    sigma = diameter / 6
    radius = max(1, int(diameter * 0.5))
    s = 2 * sigma * sigma
    ky, kx = np.ogrid[-radius:+radius+1, -radius:+radius+1]
    kernel = np.exp(-(kx * kx + ky * ky) / s)
        
    dleft, dtop = -min(x, radius), -min(y, radius)
    dright, dbottom = +min(width - x, radius+1), +min(height - y, radius+1)
    select_heatmap = heatmap[y+dtop:y+dbottom, x+dleft:x+dright]
    select_kernel = kernel[radius+dtop:radius+dbottom, radius+dleft:radius+dright]
    if min(select_heatmap.shape) > 0:
        np.maximum(select_heatmap, select_kernel, out=select_heatmap)
    return heatmap

def draw_gauss_torch(heatmap, x, y, box_size):
    if not isinstance(box_size, tuple):
        box_size = (box_size, box_size)

    box_width, box_height = box_size
    diameter = min(box_width, box_height)
    device = heatmap.device
    dtype = heatmap.dtype

    x = int(x)
    y = int(y)
    height, width = heatmap.shape[:2]
    sigma = diameter / 6
    radius = max(1, int(diameter * 0.5))
    s = 2 * sigma * sigma
    ky = torch.arange(-radius, +radius+1, device=device, dtype=dtype).view(-1, 1)
    kx = torch.arange(-radius, +radius+1, device=device, dtype=dtype).view(1, -1)
    kernel = torch.exp(-(kx * kx + ky * ky) / s)
    
    dleft, dtop = -min(x, radius), -min(y, radius)
    dright, dbottom = +min(width - x, radius+1), +min(height - y, radius+1)
    select_heatmap = heatmap[y+dtop:y+dbottom, x+dleft:x+dright]
    select_kernel = kernel[radius+dtop:radius+dbottom, radius+dleft:radius+dright]
    if min(select_heatmap.shape) > 0:
        torch.max(select_heatmap, select_kernel, out=select_heatmap)
    return heatmap

def pad_image(image, stride):
    height, width = image.shape[:2]
    pad_x = stride - (width % stride) if width % stride != 0 else 0
    pad_y = stride - (height % stride) if height % stride != 0 else 0
    image = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, (0, 0, 0))
    return image

def iou(a, b):
    ax, ay, ar, ab = a
    bx, by, br, bb = b

    cross_x = max(ax, bx)
    cross_y = max(ay, by)
    cross_r = min(ar, br)
    cross_b = min(ab, bb)
    cross_w = max(0, (cross_r - cross_x) + 1)
    cross_h = max(0, (cross_b - cross_y) + 1)
    cross_area = cross_w * cross_h
    union = (ar - ax + 1) * (ab - ay + 1) + (br - bx + 1) * (bb - by + 1) - cross_area
    return cross_area / union

def nms(bboxes, threshold, confidence_index=-1):
    bboxes.sort(key=lambda x: x[confidence_index], reverse=True)
    flags = [True] * len(bboxes)
    keep = []
    for i in range(len(bboxes)):
        if not flags[i]: continue
        keep.append(bboxes[i])

        for j in range(i+1, len(bboxes)):
            if iou(bboxes[i][:4], bboxes[j][:4]) > threshold:
                flags[j] = False
    return keep

def nmsAsClass(bboxes, threshold, class_index=-1, confidence_index=-2):

    boxasclass = {}
    for box in bboxes:
        classes = box[class_index]
        if classes not in boxasclass:
            boxasclass[classes] = []
        boxasclass[classes].append(box)

    output = []
    for key in boxasclass:
        result = nms(boxasclass[key], threshold, confidence_index)
        output.extend(result)
    return output


def iou_batch(a, b):
    # left, top, right, bottom
    a_xmin, a_xmax = a[..., 0], a[..., 2]
    a_ymin, a_ymax = a[..., 1], a[..., 3]
    b_xmin, b_xmax = b[..., 0], b[..., 2]
    b_ymin, b_ymax = b[..., 1], b[..., 3]
    inter_xmin = torch.max(a_xmin, b_xmin)
    inter_xmax = torch.min(a_xmax, b_xmax)
    inter_ymin = torch.max(a_ymin, b_ymin)
    inter_ymax = torch.min(a_ymax, b_ymax)
    inter_width = (inter_xmax - inter_xmin + 1).clamp(0)
    inter_height = (inter_ymax - inter_ymin + 1).clamp(0)
    inter_area = inter_width * inter_height

    a_width, a_height = (a_xmax - a_xmin + 1), (a_ymax - a_ymin + 1)
    b_width, b_height = (b_xmax - b_xmin + 1), (b_ymax - b_ymin + 1)
    union = (a_width * a_height) + (b_width * b_height) - inter_area
    return inter_area / union


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


def get_center_affine_transform(src_width, src_height, dst_width, dst_height):
        s = min(dst_width / src_width, dst_height / src_height)
        new_width = s * src_width
        new_height = s * src_height
        dcx = dst_width * 0.5
        dcy = dst_height * 0.5
        
        dst_points = np.array([
            [dcx - new_width * 0.5, dcy - new_height * 0.5],
            [dcx + new_width * 0.5, dcy - new_height * 0.5],
            [dcx + new_width * 0.5, dcy + new_height * 0.5],
        ], dtype=np.float32)
        
        src_points = np.array([
            [0, 0],
            [src_width, 0],
            [src_width, src_height]
        ], dtype=np.float32)
        return cv2.getAffineTransform(src_points, dst_points)

def center_affine(image, width, height):
    src_height, src_width = image.shape[:2]
    M = get_center_affine_transform(src_width, src_height, width, height)
    return cv2.warpAffine(image, M, (width, height))

def inverse_center_affine_bboxes(image_width, image_height, net_width, net_height, bboxes):
    num_bboxes = len(bboxes)
    if num_bboxes == 0:
        return bboxes
    
    M = get_center_affine_transform(image_width, image_height, net_width, net_height)
    M = np.matrix(np.vstack([M, np.array([0, 0, 1])])).I
    M = M[:2]
    
    bboxes = np.array(bboxes) # 4 x 6,   left, top, right, bottom, confidence.item(), classes.item()
    left_top = bboxes[:, :2]
    right_bottom = bboxes[:, 2:4]

    left_top_project = (M @ np.hstack([left_top, np.ones([num_bboxes, 1])]).T).T
    right_bottom_project = (M @ np.hstack([right_bottom, np.ones([num_bboxes, 1])]).T).T
    new_box = np.hstack([left_top_project, right_bottom_project, bboxes[:, 4:]])
    return new_box.tolist()

def is_parallel(model):
    # is model is parallel with DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 更快
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True

# return GFlops, MParams
def compute_flops(model, input):
    try:
        from thop import profile
        from copy import deepcopy

        #profile是侵入式的，会污染model，导致名称修改，因此需要深度复制
        flops, params = profile(deepcopy(model).eval(), inputs=input, verbose=False)
        # 单位分别是GFlops和MParams
        return flops / 1E9, params / 1E6
    except Exception as e:
        pass

    return -1, -1


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
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
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

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()



def convert_to_pixel_annotation(normalize_annotations, image_width, image_height):
    '''
    转换标注信息从normalize到pixel
    参数：
        normalize_annotations[Nx5]:   指定为标注信息，格式是[cx, cy, width, height, class_index]
        image_width[int]:             指定为标注信息的图像宽度
        image_height[int]:            指定为标注信息的图像高度
    返回值：
        pixel_annotations[Nx5]:   返回格式是[left, top, right, bottom, class_index]
    '''
    
    pixel_annotations = normalize_annotations.clone() if isinstance(normalize_annotations, torch.Tensor) else normalize_annotations.copy()
    cx, cy, width, height, class_index = [normalize_annotations[:, i] for i in range(5)]
    pixel_annotations[:, 0] = cx * image_width - (width * image_width - 1) * 0.5         # left
    pixel_annotations[:, 1] = cy * image_height - (height * image_height - 1) * 0.5      # top
    pixel_annotations[:, 2] = cx * image_width + (width * image_width - 1) * 0.5         # right
    pixel_annotations[:, 3] = cy * image_height + (height * image_height - 1) * 0.5      # bottom
    return pixel_annotations


def convert_to_normalize_annotation(pixel_annotations, image_width, image_height):
    '''
    转换标注信息到normalize格式，除以图像宽高进行归一化
    参数：
        pixel_annotations[Nx5]:       指定为标注信息，格式是[left, top, right, bottom, class_index]
        image_width[int]:             指定为标注信息的图像宽度
        image_height[int]:            指定为标注信息的图像高度
    返回值：
        normalize_annotations[Nx5]:   返回格式是[cx, cy, width, height, class_index]
    '''
    
    normalize_annotations = pixel_annotations.clone() if isinstance(pixel_annotations, torch.Tensor) else pixel_annotations.copy()
    left, top, right, bottom, class_index = [pixel_annotations[:, i] for i in range(5)]
    normalize_annotations[:, 0] = (left + right) * 0.5 / image_width  # cx
    normalize_annotations[:, 1] = (top + bottom) * 0.5 / image_height  # cy
    normalize_annotations[:, 2] = (right - left + 1) / image_width      # width
    normalize_annotations[:, 3] = (bottom - top + 1) / image_height      # height
    return normalize_annotations


def load_voc_annotation(annotation_file, label_map):
    '''
    加载标注文件xml，读取其中的bboxes
    参数：
        annotation_file[str]：  指定为xml文件路径
        label_map[list]：       指定为标签数组
    返回值：
        np.array([(xmin, ymin, xmax, ymax, class_index), (xmin, ymin, xmax, ymax, class_index)])
    '''
    with open(annotation_file, "r") as f:
        annotation_data = f.read()

    def middle(s, begin, end, pos_begin = 0):
        p = s.find(begin, pos_begin)
        if p == -1:
            return None, None

        p += len(begin)
        e = s.find(end, p)
        if e == -1:
            return None, None

        return s[p:e], e + len(end)

    obj_bboxes = []
    object_, pos_ = middle(annotation_data, "<object>", "</object>")
    while object_ is not None:
        xmin = int(middle(object_, "<xmin>", "</xmin>")[0])
        ymin = int(middle(object_, "<ymin>", "</ymin>")[0])
        xmax = int(middle(object_, "<xmax>", "</xmax>")[0])
        ymax = int(middle(object_, "<ymax>", "</ymax>")[0])
        name = middle(object_, "<name>", "</name>")[0]
        object_, pos_ = middle(annotation_data, "<object>", "</object>", pos_)
        obj_bboxes.append((xmin, ymin, xmax, ymax, label_map.index(name)))
    
    # 创建一个0 x 5的ndarray，可以使得后面的计算顺利进行，不必处理box为0时候的问题，也不会造成shape不匹配的错误
    return_ndarray_bboxes = np.zeros((0, 5), dtype=np.float32)
    if len(obj_bboxes) > 0:
        return_ndarray_bboxes = np.array(obj_bboxes, dtype=np.float32)
    return return_ndarray_bboxes


def fit_anchor(all_box_size, default_anchor, anchor_t=4, genetic_algorithm_iters=1000):
    '''
    根据情况拟合anchor

    参数：
        all_box_size[np, Nx2]:   要求是N个框的宽高，像素单位的，图像下的像素单位
        default_anchor[np, Kx2]: 要求是K个anchor的宽高，像素单位的，图像下的像素单位
    返回值：
        返回拟合后的anchor[Kx2]
    '''

    def fitness(box_wh, anchor, anchor_t):
        ratio = box_wh[:, None] / anchor[None]
        box_div_anchor = ratio
        anchor_div_box = 1 / ratio
        min_ratio = np.maximum(box_div_anchor, anchor_div_box).max(axis=2)

        # min_ratio -> N x K
        # 取每个box对9个anchor匹配度最好的那个，也就是比例差距最小的
        min_ratio = min_ratio.min(axis=1)
        return ((1 / min_ratio) * (min_ratio < anchor_t)).mean()

    def best_possible_recall(box_wh, anchor, anchor_t):
        # box_wh[Nx2]
        # anchor[Kx2]
        # box_wh[:, None] -> box_wh.shape = Nx1x2
        # anchor[None]    -> anchor.shape = 1xKx2
        # ratio.shape = NxKx2
        ratio = box_wh[:, None] / anchor[None]
        box_div_anchor = ratio
        anchor_div_box = 1 / ratio
        min_ratio = np.maximum(box_div_anchor, anchor_div_box).max(axis=2)

        # min_ratio -> N x K
        # 取每个box对9个anchor匹配度最好的那个
        min_ratio = min_ratio.min(axis=1)
        return (min_ratio < anchor_t).mean()

    default_anchor_bpr = best_possible_recall(all_box_size, default_anchor, anchor_t)

    if default_anchor_bpr >= 0.99:
        print(f"Default anchor bpr is {default_anchor_bpr:.5f}")
        return default_anchor

    num_anchor = default_anchor.shape[0]
    box_std = all_box_size.std(axis=0)
    norm_box_size = all_box_size / box_std
    current_anchor, _ = kmeans(norm_box_size, num_anchor, iter=30)
    current_anchor = current_anchor * box_std
    current_fitness = fitness(all_box_size, current_anchor, anchor_t)

    anchor_shape = current_anchor.shape
    pbar = tqdm(range(genetic_algorithm_iters))

    # 使用遗传算法迭代变异anchor，直到找到更加优秀的anchor
    for _ in pbar:
        v = np.ones(anchor_shape)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)，变异，直到发生变换，避免重复
            v = ((np.random.random(anchor_shape) < 0.9) * np.random.random() * np.random.randn(*anchor_shape) * 0.1 + 1).clip(min=0.3, max=3.0)
        
        # anchor不能小于2
        mutate_anchor = (current_anchor * v).clip(min=2.0)
        mutate_fitness = fitness(mutate_anchor, all_box_size, anchor_t)
        pbar.set_description(f'Evolving anchors with Genetic Algorithm: fitness = {mutate_fitness:.4f} / {current_fitness:.4f}')
        
        if mutate_fitness > current_fitness:
            current_anchor = mutate_anchor
            current_fitness = mutate_fitness

    current_bpr = best_possible_recall(all_box_size, current_anchor, anchor_t)
    if current_bpr > default_anchor_bpr:
        print(f"Current bpr[{current_bpr:.5f}] > Default bpr[{default_anchor_bpr}]")
    else:
        current_anchor = default_anchor
        print(f"Current bpr[{current_bpr:.5f}] < Default bpr[{default_anchor_bpr}]")

    # 对anchor使用面积进行排序
    return current_anchor[current_anchor.prod(axis=1).argsort()]


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(conv.in_channels,
                            conv.out_channels,
                            kernel_size=conv.kernel_size,
                            stride=conv.stride,
                            padding=conv.padding,
                            bias=True).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        return fusedconv