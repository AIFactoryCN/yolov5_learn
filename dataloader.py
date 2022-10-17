from email import utils
from json import load
from unittest.mock import patch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import glob
import numpy as np
from util import *
import cv2
from PIL import Image, ExifTags, ImageOps
import contextlib
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def createDataLoader(path, imgSize, batchSize, augment):
    dataSet = MyDataSet(path, imgSize=imgSize, augment=augment)
    batchSize = min(batchSize, len(dataSet))
    loader = DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=True, collate_fn=MyDataSet.collate_fn)
    return loader, dataSet

def exif_size(img):
    # 获取图片中旋转信息tag对应的key
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    # Returns exif-corrected PIL size
    # TODO 补充关于exif相关信息
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s   

class MyDataSet(Dataset):
    def __init__(self, path, imgSize=640, augment=False, image_dir_name='JPEGImages', annotation_dir_name='labels', anno_suffix='txt'):
        self.path = path

        self.imgSize = imgSize
        self.augment = augment
        files = []
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)
            if p.is_dir():
                files += glob.glob(str(p / '**' / '*.*'), recursive=True)
            elif p.is_file():
                with open(p) as f:
                    f = f.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    files += [x.replace('./', parent) if x.startswith('./') else x + '.jpg' for x in f]
        self.img_files = sorted(x.replace('/', os.sep) for x in files if x.split('.')[-1].lower() in IMG_FORMATS)
        assert self.img_files, f'No images data found'
        sImg, sAnno = f'{os.sep}{image_dir_name}{os.sep}', f'{os.sep}{annotation_dir_name}{os.sep}'
        self.label_files = [sAnno.join(x.rsplit(sImg, 1)).rsplit('.', 1)[0] + f'.{anno_suffix}' for x in self.img_files]
        self.verify_images_labels()

    def verify_images_labels(self):
        # data_dict: {图片路径: 标注信息}
        data_dict = {}
        #TODO
        n_miss, n_found, n_empty, n_corrupt, msg = 0, 0, 0, 0, ""
        for img_file, anno_file in zip(self.img_files, self.label_files):
            try:
                # 校验图片
                img = Image.open(img_file)
                img.verify()  # PIL verify
                # 根据exif中旋转信息，获取正确的图像shape
                shape = exif_size(img)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert img.format.lower() in IMG_FORMATS, f'invalid image format {img.format}'
                if img.format.lower() in ('jpg', 'jpeg'):
                    with open(img_file, 'rb') as f:
                        '''
                        f.seek():
                        seek() 方法用于移动文件读取指针到指定位置
                        参数1: offset, 偏移的字节数, 若为负数，则从倒数第几位开始
                        参数2: whence, 0 代表从文件开头开始算起，1 代表从当前位置开始算起，2 代表从文件末尾算起。默认为0
                        举例:
                            0123456789
                            f.seek(3): 012|3456789
                            f.seek(-2, 2): 01234567|89
                        '''
                        f.seek(-2, 2)
                        '''
                        f.read() != b'\xff\xd9'：用于检测错误jpeg图片
                        何为错误图片？当存在网络问题或存储不当，可能会导致图片存储不完整，如一张图片上只有一半有画面
                        如何检测？有种较好的方式就是判断图片结尾的标识，
                            JPG文件结尾标识  \xff\xd9
                            JPEG文件结尾标识 \xff\xd9
                            PNG文件结尾标识  \xaeB`\x82
                        此处，就通过判断jpg/jpeg后两位的文件结尾标识，判断图片是否错误
                        总结：通过f.seek(-2, 2)读取文件的最后两个字节判断是否为正常的图片结尾标识，来清洗我们的数据集中的错误数据
                        '''
                        if f.read() != b'\xff\xd9':  # corrupt JPEG
                            # TODO 补充关于 exif_transpose的信息
                            ImageOps.exif_transpose(Image.open(img_file)).save(img_file, 'JPEG', subsampling=0, quality=100)
                            msg = f'WARNING ⚠️ {img_file}: corrupt JPEG restored and saved'

                # 校验标注
                if os.path.isfile(anno_file):
                    with open(anno_file) as f:
                        # [ [class1, cx, cy, width, height],
                        #   [class2, cx, cy, width, height] ]
                        lb = [x.split() for x in f.read().strip().splitlines() if len(x)] # 2维
                        lb = np.array(lb, dtype=np.float32)
                    if len(lb):
                        # 检查标注是否有5个元素
                        assert lb.shape[1] == 5, f"labels require 5 elements, but {lb.shape[1]} detected"
                        # 检查标注是否有效(>=0)
                        # .all() 表示是否全都满足要求，若不满足，则为false
                        assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                        # 检查标注是否都已经做了归一化操作
                        assert (lb[..., 1:] <= 1).all(), f"non-normalized or out of bounds coordinates {lb[..., 1:][lb[..., 1:] > 1]}"
                        # 去掉重复的标注框 np.unique
                        '''
                        np.unique方法：
                        参数：
                            ar : array_like
                                输入数组，若不设置axis参数或设置axis参数为None时，会把数组展开为1维数组进行操作
                            return_index : bool, optional
                                如果设置为True，则返回新数组在输入数组内的索引，输入数组[索引]=新数组
                            return_inverse : bool, optional
                                如果设置为True，则返回输入数组在新数组内的索引，新数组[索引]=输入数组
                            return_counts : bool, optional
                                如果设置为True，返回新数组内元素在输入数组内的个数
                            axis : int or None, optional
                                维度信息
                        官方示例:
                            >>> np.unique([1, 1, 2, 2, 3, 3])
                            array([1, 2, 3])
                            >>> a = np.array([[1, 1], [2, 3]])
                            >>> np.unique(a)
                            array([1, 2, 3])

                            >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
                            >>> np.unique(a, axis=0)
                            array([[1, 0, 0], [2, 3, 4]])

                            >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
                            >>> u, indices = np.unique(a, return_index=True)
                            >>> u
                            array(['a', 'b', 'c'], dtype='<U1')
                            >>> indices
                            array([0, 1, 3])
                            >>> a[indices]
                            array(['a', 'b', 'c'], dtype='<U1')

                            >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
                            >>> u, indices = np.unique(a, return_inverse=True)
                            >>> u
                            array([1, 2, 3, 4, 6])
                            >>> indices
                            array([0, 1, 4, 3, 1, 2, 1])
                            >>> u[indices]
                            array([1, 2, 6, 4, 2, 3, 2])

                            >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
                            >>> values, counts = np.unique(a, return_counts=True)
                            >>> values
                            array([1, 2, 3, 4, 6])
                            >>> counts
                            array([1, 3, 1, 1, 1])
                            >>> np.repeat(values, counts)
                            array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved
                        '''
                        _, indices = np.unique(lb, axis=0, return_index=True)
                        # 检查是否有重复，若有重复，使用去重后的标注
                        if len(indices) < len(lb):
                            # 输入数组取索引得到非重复的标注
                            lb = lb[indices]
                            msg = f'WARNING ⚠️ {img_file}: {len(lb) - len(indices)} duplicate labels removed'
                    else:
                        # 若本张数据有标注文件但标注文件内没有标注，则说明该图片没有需要训练的所有分类，标注信息置为0
                        lb = np.zeros((0, 5), dtype=np.float32)
                else:
                    # 若本张数据没有标注文件，则说明该图片没有需要训练的所有分类，标注信息置为0
                    lb = np.zeros((0, 5), dtype=np.float32)
                data_dict[img_file] = lb
            except Exception as e:
                # 若存在异常，则不取用该数据
                msg = f'WARNING ⚠️ {img_file}: ignoring corrupt image/label: {e}'

        self.img_files = list(data_dict.keys())
        self.labels = list(data_dict.values())
    
    def xywhn2xyxy(self, x, w=640, h=640, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
        return y
    
    def xyxy2xywhn(self, x, w=640, h=640, eps=0.0):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
        y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
        y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
        return y

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        f = self.img_files[i]

        image = cv2.imread(f)  # BGR
        assert image is not None, f'Image Not Found {f}'
        image_height, image_width = image.shape[:2]  # orig hw
        r = self.imgSize / max(image_height, image_width)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            image = cv2.resize(image, (int(image_width * r), int(image_height * r)), interpolation=interp)
        return image, (image_height, image_width), image.shape[:2]  # im, hw_original, hw_resized

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img, (h0, w0), (h, w) = self.load_image(index)
        shape = self.imgSize
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        labels = self.labels[index].copy()
        labels[:, 1:] = self.xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        num_labels = len(labels)
        if num_labels:
            labels[:, 1:5] = self.xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], eps=1E-3) 
            
        labels_out = torch.zeros((num_labels, 6))
        # [ [0, class1, cx, cy, width, height],
        #   [0, class2, cx, cy, width, height] ]
        labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths, shapes = zip(*batch)  # transposed
        for i, label in enumerate(labels):
            label[:, 0] = i  # add target image index for build_targets()
        # labels.shape [image_index, class_index, x1, y1, x2, y2]
        return torch.stack(imgs, 0), torch.cat(labels, 0), paths, shapes



if __name__ == '__main__':
    path = "/home/bai/bai/sourceCode/baiCode/yolo-rewrite/testPath/a.txt"
    path = Path(path)
    print(path.parent)