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
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def createDataLoader(path, imgSize, batchSize, augment):
    dataSet = MyDataSet(path, imgSize=imgSize, augment=augment)
    batchSize = min(batchSize, len(dataSet))
    loader = DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=True, collate_fn=MyDataSet.collate_fn)
    return loader, dataSet
    

class MyDataSet(Dataset):
    def __init__(self, path, imgSize=640, augment=False, imgDirName='images', annoDirName='labels', annoSuffix='txt'):
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
                    files += [x.replace('./', parent) if x.startswith('./') else x for x in f]
        self.imgFiles = sorted(x.replace('/', os.sep) for x in files if x.split('.')[-1].lower() in IMG_FORMATS)
        assert self.imgFiles, f'No images data found'
        sImg, sAnno = f'{os.sep}{imgDirName}{os.sep}', f'{os.sep}{annoDirName}{os.sep}'
        self.labelFiles = [sAnno.join(x.rsplit(sImg, 1)).rsplit('.', 1)[0] + f'.{annoSuffix}' for x in self.imgFiles]
        self.verifyImgsLabels()
    def verifyImgsLabels(self):
        dataDict = {}
        #TODO
        nMiss, nFound, nEmpty, nCorrupt, msgs = 0, 0, 0, 0, []
        for imgFile, annoFile in zip(self.imgFiles, self.labelFiles):
            
            #TODO
            #im = Image.open(im_file)

            if os.path.isfile(annoFile):
                with open(annoFile) as f:
                    # [ [class1, x1, y1, x2, y2],
                    #   [class2, x1, y1, x2, y2] ]
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)] # 2维
                    # # [class1, class2]
                    # classes = np.array([x[0] for x in lb], dtype=np.float32)
                    # # [
                    # #   [[x1, y1],
                    # #    [x2, y2]],
                    # #   [[x1, y1],
                    # #    [x2, y2]]
                    # # ]
                    # segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    # lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                    lb = np.array(lb, dtype=np.float32)
                    if not len(lb):
                        # lb = np.zeros((0, 5), dtype=np.float32)
                        continue
                dataDict[imgFile] = lb

        self.imgFiles = list(dataDict.keys())
        self.labels = list(dataDict.values())

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        f = self.imgFiles[i]

        im = cv2.imread(f)  # BGR
        assert im is not None, f'Image Not Found {f}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.imgSize / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, index):
        img, (h0, w0), (h, w) = self.load_image(index)
        shape = self.imgSize
        img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        labels = self.labels[index].copy()
        num_labels = len(labels) 
        labels_out = torch.zeros((num_labels, 6))
        labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        return torch.from_numpy(img), labels_out, self.imgFiles[index], shapes

    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths, shapes = zip(*batch)  # transposed
        for i, label in enumerate(labels):
            label[:, 0] = i  # add target image index for build_targets()
        return torch.stack(imgs, 0), torch.cat(labels, 0), paths, shapes

if __name__ == '__main__':
    path = "/home/bai/bai/sourceCode/baiCode/yolo-rewrite/testPath/a.txt"
    path = Path(path)
    print(path.parent)