import os
import sys
import cv2
import torch
import random
import sys_utils
import nn_utils

import numpy as np
import torch.nn as nn
import data_provider

from sys_utils import _single_instance_logger as logger
from PIL import Image

# 格式定义：
# pixel_annotations：       像素为单位的标注，格式是[left, top, right, bottom]，绝对位置标注
# normalize_annotations:    归一化后的标注0-1，除以图像宽高，格式是[cx, cy, width, height]

# 总结：
# 1.hsv增广没做v
# 2.exif信息没用到
# 3.collate_fn函数没测试
# 4.随机翻转v

class Dataset:
    def __init__(self, augment, image_size, provider, batch_size=None, max_stride=32):
        self.augment = augment
        self.image_size = image_size
        self.border_fill_value = 114
        self.provider = provider

        # 获取所有标注的尺寸信息，基于像素宽高
        # all_labeled_information is   **[ jpeg_file, normalize_annotations, [image_width, image_height] ]**
        # normalize_annotations is **[ [cx, cy, width, height, class] ] ndarray**
        all_annotation_sizes = []
        for item in self.provider.all_labeled_information:
            normalize_annotations = item[1]
            item_image_size = item[2]
            
            annotation_size = normalize_annotations[:, [2, 3]] * item_image_size
            all_annotation_sizes.append(annotation_size)
        self.all_annotation_sizes = np.concatenate(all_annotation_sizes, axis=0)

        if not self.augment:
            # build rectangular
            batch_shapes_wh, batch_index_list, sorted_index = self.build_rectangular(image_size, batch_size, max_stride, provider.all_labeled_information)

            # 得到批次的形状，以及根据索引重新编排咱们的数据
            self.batch_shapes_wh    = batch_shapes_wh
            self.batch_index_list   = batch_index_list
            self.provider.all_labeled_information = [self.provider.all_labeled_information[i] for i in sorted_index]

    
    def build_rectangular(self, image_size, batch_size, max_stride, all_labeled_information, pad=0.5):
        '''
        Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        说明：
            构建基于矩形训练的信息。其含义有：
            1. dataset不能够使用shuffle的方式进行数据获取
            2. 对于所有图像集合S(image, size[Nx2, w h])，训练批次为B
            3. 排序S，基于size的宽高比，得到I
            4. 使得每个批次获取的图像，其宽高比都是接近的。因此可以直接对所有图进行最小填充，使得宽高一样作为输出。这样我们就得到一批一样大小的图进行迭代了
                但是每个批次间，可能会不一样
        '''

        # all_labeled_information is   **[ jpeg_file, normalize_annotations, [image_width, image_height] ]**
        datas = all_labeled_information
        total_images = len(datas)

        # 比如有10个图，batch=3
        # 得到的是：0001112223
        batch_index_list = np.floor(np.arange(total_images) / batch_size).astype(np.int32)

        # 没轮迭代所需要的批次数
        number_of_batches = batch_index_list[-1] + 1

        # 提取所有的宽高尺寸
        shapes_wh = np.array(list(map(lambda x: x[2], datas)), dtype=np.float32)

        # 计算高宽比
        aspect_ratio = shapes_wh[:, 1] / shapes_wh[:, 0]

        # 排序并得到索引
        index_of_image_based_aspect_ratio = aspect_ratio.argsort()

        # 重新编序宽高比
        aspect_ratio = aspect_ratio[index_of_image_based_aspect_ratio]

        # 计算每个shape基于image_size的宽高比
        batch_shapes_wh = [[1, 1]] * number_of_batches
        for ibatch in range(number_of_batches):

            # 提取属于这个batch的所有宽高比（横纵比）
            aspect_ratio_of_ibatch = aspect_ratio[batch_index_list == ibatch]

            # 计算最小和最大的高宽比
            min_aspect_ratio, max_aspect_ratio = aspect_ratio_of_ibatch.min(), aspect_ratio_of_ibatch.max()

            if max_aspect_ratio < 1:
                # 如果这个批次的图像全都是，宽度小于高度的。说明长边是高度方向
                # 长边设置为图像大小，短边则用宽高比
                batch_shapes_wh[ibatch] = [1, max_aspect_ratio]
            elif min_aspect_ratio > 1:
                # 如果这个批次的图像全都是，宽度大于高度的
                batch_shapes_wh[ibatch] = [1 / min_aspect_ratio, 1]

        batch_shapes_wh = np.ceil(np.array(batch_shapes_wh) * image_size / max_stride + pad).astype(np.int) * max_stride
        return batch_shapes_wh, batch_index_list, index_of_image_based_aspect_ratio


    def __len__(self):
        return len(self.provider)


    def __getitem__(self, image_indice):
        if self.augment:
            image, normalize_annotations = self.load_mosaic(image_indice)
            restore_info = None

            # 应用hsv增广
            image = self.hsv_augment(image)

            # 一定概率应用随机水平翻转
            if random.random() < 0.5:
                image, normalize_annotations = self.horizontal_flip(image, normalize_annotations)
        else:
            batch_index = self.batch_index_list[image_indice]
            batch_width, batch_height = self.batch_shapes_wh[batch_index]
            image, normalize_annotations, restore_info = self.load_center_affine(image_indice, batch_width, batch_height)

        num_targets = len(normalize_annotations)

        # BGR to RGB
        bgr_image = image
        image = np.ascontiguousarray(bgr_image[..., ::-1].transpose(2, 0, 1)) 

        # normalize
        image = image / np.array([255.0], dtype=np.float32)

        # image_id, class_index, cx, cy, width, height
        output_annotations = np.zeros((num_targets, 6), dtype=np.float32)
        output_annotations[:, 1:] = normalize_annotations[:, [4, 0, 1, 2, 3]]
        return torch.from_numpy(image), bgr_image, torch.from_numpy(output_annotations), normalize_annotations, restore_info


    def load_center_affine(self, image_indice, output_width, output_height):
        '''
        加载图像，采用中心对齐的方式
        返回值：
            image, normalize_annotations
        '''

        image, normalize_annotations, (width, height), (origin_width, origin_height), scale = self.load_image_with_uniform_scale(image_indice)
        #nn_utils.draw_norm_bboxes(image, normalize_annotations, color=(0, 0, 255), thickness=5)

        pad_width = output_width - width
        pad_height = output_height - height

        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        fill_value = (self.border_fill_value, self.border_fill_value, self.border_fill_value)
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=fill_value)

        # 框处理
        # (norm_cx * origin_width + pad_left) / self.image_size
        # norm_cx * origin_width / self.image_size + pad_left / self.image_size
        #        alpha =  origin_width / self.image_size
        #        beta =   pad_left / self.image_size
        #
        # norm_width * origin_width / self.image_size
        # norm_width * alpha

        x_alpha = width / output_width
        x_beta = pad_left / output_width
        y_alpha = height / output_height
        y_beta = pad_top / output_height

        # cx, cy
        normalize_annotations[:, [0, 1]] = normalize_annotations[:, [0, 1]] * [x_alpha, y_alpha] + [x_beta, y_beta]

        # width, height
        normalize_annotations[:, [2, 3]] = normalize_annotations[:, [2, 3]] * [x_alpha, y_alpha]
        return image, normalize_annotations, (pad_left, pad_top, origin_width, origin_height, scale)


    # def horizontal_flip(self, image, normalize_annotations):

    #     image = cv2.flip(image, 1)

    #     # cx, cy, width, height, class_index
    #     normalize_annotations[:, 0] = 1 - normalize_annotations[:, 0]
    #     return image, normalize_annotations
    def horizontal_flip(self, image, normalize_annotations):
        '''
        对图像和框进行水平翻转
        参数：
            image：提供图像
            normalize_annotations：提供归一化后的框信息，格式是[cx, cy, width, height, class_index]
        返回值：
            image, normalize_annotations
        '''
        
        # flipCode = 1 ，   水平，也就是x轴翻转
        # flipCode = 0，    垂直，也就是y轴翻转
        # flipCode = -1，   对角翻转，x和y都发生翻转
        image = cv2.flip(image, flipCode=1)
        #image = np.fliplr(image)
        normalize_annotations = normalize_annotations.copy()

        # cx, cy, width, height
        # 0-1
        # (image_width - 1) / image_width
        image_width = image.shape[1]  # Height, Width, Channel
        normalize_annotations[:, 0] = (image_width - 1) / image_width - normalize_annotations[:, 0]
        return image, normalize_annotations


    def hsv_augment(self, image, hue_gain=0.015, saturation_gain=0.7, value_gain=0.4):
        '''
        对图像进行HSV颜色空间增广
        参数：
            hue_gain:          色调增益，最终增益系数为  random(-1, +1) * hue_gain + 1
            saturation_gain:   饱和度增益，最终增益系数为  random(-1, +1) * saturation_gain + 1
            value_gain:        亮度增益，最终增益系数为  random(-1, +1) * value_gain + 1
        返回值：
            image
        '''

        # 把增益值修改为最终的增益系数
        hue_gain = np.random.uniform(-1, +1) * hue_gain + 1
        saturation_gain = np.random.uniform(-1, +1) * saturation_gain + 1
        value_gain = np.random.uniform(-1, +1) * value_gain + 1

        # 把图像转换为HSV后并分解为H、S、V，3个通道
        hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        # cv2.COLOR_BGR2HSV       ->  相对压缩过的，可以说是有点损失的
        # cv2.COLOR_BGR2HSV_FULL  ->  完整的

        # hue        ->  值域 0 - 179
        # saturation ->  值域 0 - 255
        # value      ->  值域 0 - 255

        # LUT，look up table
        # table  -> [[10, 255, 7], [255, 0, 0], [0, 255, 0]]
        # index  -> [2, 0, 1]
        # value  -> [[0, 255, 0], [10, 255, 7], [255, 0, 0]]

        dtype = image.dtype
        lut_base = np.arange(0, 256)
        lut_hue = ((lut_base * hue_gain) % 180).astype(dtype)
        lut_saturation = np.clip(lut_base * saturation_gain, 0, 255).astype(dtype)
        lut_value = np.clip(lut_base * value_gain, 0, 255).astype(dtype)

        # cv2.LUT(index, lut)
        changed_hue = cv2.LUT(hue, lut_hue)
        changed_saturation = cv2.LUT(saturation, lut_saturation)
        changed_value = cv2.LUT(value, lut_value)

        image_hsv = cv2.merge((changed_hue, changed_saturation, changed_value))
        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)



    def load_mosaic(self, image_indice):
        '''
        加载图像，并使用马赛克增广
            - 先把1个image_indice指定的图，和其他3个随机图，拼接为马赛克
            - 使用随机仿射变换，输出image_size大小
            - 移除无效的框
            - 恢复框为normalize
        返回值：
            image[self.image_size x self.image_size], normalize_annotations
        '''
        
        # 在image_size * 0.5到image_size * 1.5之间随机一个中心
        # 马赛克的第一步是拼接为一个大图，即image_size * 2, image_size * 2
        x_center = int(random.uniform(self.image_size * 0.5, self.image_size * 1.5))
        y_center = int(random.uniform(self.image_size * 0.5, self.image_size * 1.5))

        num_images = len(self.provider)
        all_image_indices = [image_indice] + [random.randint(0, num_images - 1) for _ in range(3)]

        #  img1,  img2
        #  img3,  img4
        alignment_corner_point = [
            [1, 1],   # img1的角点相对于其宽高尺寸的位置
            [0, 1],   # img2的角点相对于其宽高尺寸的位置
            [1, 0],   # img3的角点相对于其宽高尺寸的位置
            [0, 0]    # img4的角点相对于其宽高尺寸的位置
        ]

        merge_mosaic_image_size = self.image_size * 2
        merge_mosaic_image = np.full((merge_mosaic_image_size, merge_mosaic_image_size, 3), self.border_fill_value, dtype=np.uint8)
        merge_mosaic_pixel_annotations = []

        for index, (image_indice, (corner_point_x, corner_point_y)) in enumerate(zip(all_image_indices, alignment_corner_point)):

            image, normalize_annotations, (image_width, image_height), (origin_width, origin_height), scale = self.load_image_with_uniform_scale(image_indice)
            corner_point_x = corner_point_x * image_width
            corner_point_y = corner_point_y * image_height

            x_offset = x_center - corner_point_x   # 先加目标图的x，再减去image上的x
            y_offset = y_center - corner_point_y

            M = np.array([
                [1, 0, x_offset],
                [0, 1, y_offset]
            ], dtype=np.float32)

            cv2.warpAffine(image, M, (merge_mosaic_image_size, merge_mosaic_image_size), 
                dst=merge_mosaic_image, 
                borderMode=cv2.BORDER_TRANSPARENT,
                flags=cv2.INTER_NEAREST)

            # left_small = max(-x_offset, 0)
            # top_small = max(-y_offset, 0)
            # right_small = image_width - max((x_offset + image_width) - merge_mosaic_image_size, 0)
            # bottom_small = image_height - max((y_offset + image_height) - merge_mosaic_image_size, 0)
            # left_large = max(x_offset, 0)
            # top_large = max(y_offset, 0)
            # right_large = min(x_offset + image_width, merge_mosaic_image_size)
            # bottom_large = min(y_offset + image_height, merge_mosaic_image_size)
            # merge_mosaic_image[top_large:bottom_large, left_large:right_large] = image[top_small:bottom_small, left_small:right_small]

            # 把框转换为像素单位，并且是[left, top, right, bottom, class_index]格式的
            pixel_annotations = nn_utils.convert_to_pixel_annotation(normalize_annotations, image_width, image_height)

            # 把框进行平移
            pixel_annotations = pixel_annotations + [x_offset, y_offset, x_offset, y_offset, 0]

            # 把所有框合并
            merge_mosaic_pixel_annotations.append(pixel_annotations)
            
        # 所有框拼接为一个矩阵
        merge_mosaic_pixel_annotations = np.concatenate(merge_mosaic_pixel_annotations, axis=0)

        # 如果框越界了，需要限制到范围内，inplace操作
        np.clip(merge_mosaic_pixel_annotations[:, :4], a_min=0, a_max=merge_mosaic_image_size-1, out=merge_mosaic_pixel_annotations[:, :4])

        # 随机仿射变换
        scale = random.uniform(0.5, 1.5)

        # 随机仿射变化
        #  1. 进行缩放
        #  2. 将large图的中心移动到，目标图(small)的中心上
        #       - large -> 1280 x 1280
        #       - small ->  640 x 640
        # 也知道中心是：self.image_size, self.image_size，缩放系数是scale
        #    x * M00 + 0 * m01 + x_offset    
        #    ix * M00 + x_offset  =  dx
        #       已知一个解，指的是中心点的解
        #       large.center.x * M00 + x_offset = small.center.x
        #       x_offset = small.center.x - large.center.x * M00
        #                = small.center.x - large.center.x * scale
        #                = image_size * 0.5 - image_size * scale
        #                = image_size * (0.5 - scale)
        #   
        #    指定的中心，有什么样的定义：
        M = np.array([
            [scale, 0, self.image_size * (0.5 - scale)],
            [0, scale, self.image_size * (0.5 - scale)]
        ], dtype=np.float32)

        merge_mosaic_image = cv2.warpAffine(merge_mosaic_image, M, 
            (self.image_size, self.image_size), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(self.border_fill_value, self.border_fill_value, self.border_fill_value))
        
        # 使用M矩阵对框进行变换，达到目标位置
        num_targets = len(merge_mosaic_pixel_annotations)
        output_normalize_annotations = np.zeros((0, 5))

        if num_targets > 0:
            # 映射标注框到目标图，使用M矩阵
            targets_temp = np.ones((num_targets * 2, 3))

            # 内存排布知识
            # N x 5
            # N x 4 -> left, top, right, bottom, left, top, right, bottom, left, top, right, bottom, 
            #       -> reshape(N x 2, 2)
            #       -> left, top
            #       -> right, bottom
            #       -> left, top
            #       -> right, bottom
            # 把box标注信息变成一行一个点
            targets_temp[:, :2] = merge_mosaic_pixel_annotations[:, :4].reshape(num_targets * 2, 2)

            # targets_temp ->  2N x 3
            # M -> 2 x 3
            # output: 2N x 2,  
            merge_projection_pixel_annotations = merge_mosaic_pixel_annotations.copy()
            merge_projection_pixel_annotations[:, :4] = (targets_temp @ M.T).reshape(num_targets, 4)

            # 处理框
            # 1. 裁切到图像范围
            # 2. 过滤掉无效的框
            np.clip(merge_projection_pixel_annotations[:, :4], a_min=0, a_max=self.image_size-1, out=merge_projection_pixel_annotations[:, :4])

            # 过滤无效的框
            projection_box_width = merge_projection_pixel_annotations[:, 2] - merge_projection_pixel_annotations[:, 0] + 1
            projection_box_height = merge_projection_pixel_annotations[:, 3] - merge_projection_pixel_annotations[:, 1] + 1
            original_box_width = merge_mosaic_pixel_annotations[:, 2] - merge_mosaic_pixel_annotations[:, 0] + 1
            original_box_height = merge_mosaic_pixel_annotations[:, 3] - merge_mosaic_pixel_annotations[:, 1] + 1

            area_projection = projection_box_width * projection_box_height
            area_original = original_box_width * original_box_height

            aspect_ratio = np.maximum(projection_box_width / (projection_box_height + 1e-6), projection_box_height / (projection_box_width + 1e-6))

            # 保留的条件分析
            # 1. 映射后的框，宽度必须大于2
            # 2. 映射后的框，高度必须大于2
            # 3. 裁切后的面积 / 裁切前的面积 > 0.2
            # 4. max(宽高比，高宽比) < 20
            keep_indices = (projection_box_width > 2) & \
                           (projection_box_height > 2) & \
                           (area_projection  / (area_original * scale + 1e-6) > 0.2) & \
                           (aspect_ratio < 20)
            merge_projection_pixel_annotations = merge_projection_pixel_annotations[keep_indices]
            output_normalize_annotations = nn_utils.convert_to_normalize_annotation(merge_projection_pixel_annotations, self.image_size, self.image_size)

        return merge_mosaic_image, output_normalize_annotations


    def load_image_with_uniform_scale(self, image_indice):
        '''
        加载图像，并进行长边等比缩放到self.image_size大小
        返回值：
            image, normalize_annotations, (image_resized_width, image_resized_height)
        '''
        image, normalize_annotations, (image_width, image_height) = self.provider[image_indice]
        to_image_size_ratio = self.image_size / max(image_width, image_height)

        if to_image_size_ratio != 1:

            # 如果不需要增广（评估阶段），并且缩放系数小于1，就使用效果比较好的插值方式
            if to_image_size_ratio < 1 and not self.augment:
                interp = cv2.INTER_AREA     # 速度慢，效果好，区域插值
            else:
                interp = cv2.INTER_LINEAR   # 速度快，效果也还ok，线性插值

            #image = cv2.resize(image, (0, 0), fx=to_image_size_ratio, fy=to_image_size_ratio, interpolation=interp)
            image = cv2.resize(image, (int(to_image_size_ratio * image_width), int(to_image_size_ratio * image_height)), interpolation=interp)
            
        image_resized_height, image_resized_width = image.shape[:2]
        return image, normalize_annotations, (image_resized_width, image_resized_height), (image_width, image_height), to_image_size_ratio


    @staticmethod
    def collate_fn(batch):
        '''
        属于dataset.__getitem__之后，dataloader获取数据之前
        获取数据之前，指：for batch_index, (images, labels) in enumerate(dataloader):
        在这里需要准备一个image_id

        因为这里预期返回的内容有：
        images[torch.FloatTensor]，normalize_annotations[Nx6][image_id, class_index, cx, cy, width, height], visual_info
        - visual_info指，给visdom用来显示的东西。返回batch中某一张图的信息，[image, annotations, image_id]
        '''
        # batch = [[image, annotations], [image, annotations]]
        images, original_images, normalize_annotations, original_normalize_annotations, restore_info = zip(*batch)
        for index, annotations in enumerate(normalize_annotations):
            annotations[:, 0] = index

        # 准备visual_info，用来显示的东西， image, annotations, image_id
        visual_image_id = random.randint(0, len(original_images)-1)
        visual_image = original_images[visual_image_id]
        visual_annotations = original_normalize_annotations[visual_image_id]
        visual_info = visual_image_id, visual_image, visual_annotations, restore_info

        normalize_annotations = torch.cat(normalize_annotations, dim=0)
        images = torch.stack(images, dim=0)
        return images, normalize_annotations, visual_info


if __name__ == "__main__":

    from tqdm import tqdm
    
    voc_provider = data_provider.VOCProvider("/data-rbd/wish/four_lesson/dataset/voc2007/VOCdevkitTest/VOC2007/")
    dataset = Dataset(False, 640, voc_provider, 32) 

    for item in tqdm(range(len(dataset))):
        val = dataset[item]