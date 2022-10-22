import torch.nn as nn
import torch
import torchvision

'''
目前先不使用该代码
'''

class Head(nn.Module):
    def __init__(self, num_classes=1, strides=None, anchors=None):
        super().__init__()

        '''
        anchors[3x3x2]   ->  level x scale x [width, height], 已经除以了stride后的anchor
        strides[list(3)] ->  [8, 16, 32]
        '''

        self.num_classes = num_classes
        self.gap_threshold = 4 # anchor_t
        # TODO 先固定方便调试
        self.num_anchor = 3
        self.strides = [8, 16, 32]
        self.anchors = torch.tensor([
            [10,13, 16,30, 33,23],
            [30,61, 62,45, 59,119],
            [116,90, 156,198, 373,326]
        ]).view(3, 3, 2) / torch.FloatTensor(self.strides).view(3, 1, 1)

    def detect(self, predict, confidence_threshold=0.3, nms_threshold=0.5, multi_table=True):
        '''
        检测目标
        参数：
        predict[layer8, layer16, layer32],      每个layer是BxCxHxW
        confidence_threshold，                  保留的置信度阈值
        nms_threshold，                         nms的阈值
        '''
        batch = predict[0].shape[0]
        device = predict[0].device
        objs = []
        for ilayer, (layer, stride) in enumerate(zip(predict, self.strides)):
            layer_height, layer_width = layer.shape[-2:]
            layer = layer.view(batch, 3, 5 + self.num_classes, layer_height, layer_width).permute(0, 1, 3, 4, 2).contiguous()
            layer = layer.sigmoid().view(batch, 3, -1, layer.size(-1))
            
            if self.num_classes == 1:
                object_score = layer[..., 4]
                object_classes = torch.zeros_like(object_score)
                keep_batch_indices, keep_anchor_indices, keep_cell_indices = torch.where(object_score > confidence_threshold)
            else:
                layer_confidence = layer[..., [4]] * layer[..., 5:]
                if multi_table:
                    keep_batch_indices, keep_anchor_indices, keep_cell_indices, object_classes = torch.where(layer_confidence > confidence_threshold)
                    object_score = layer_confidence[keep_batch_indices, keep_anchor_indices, keep_cell_indices, object_classes]
                else:
                    object_score, object_classes = layer_confidence.max(-1)
                    keep_batch_indices, keep_anchor_indices, keep_cell_indices = torch.where(object_score > confidence_threshold)
            
            num_keep_box = len(keep_batch_indices)
            if num_keep_box == 0:
                continue

            keepbox = layer[keep_batch_indices, keep_anchor_indices, keep_cell_indices].float()
            layer_anchors = self.anchors[ilayer]
            keep_anchors = layer_anchors[keep_anchor_indices]
            cell_x = keep_cell_indices % layer_width
            cell_y = keep_cell_indices // layer_width
            keep_cell_xy = torch.cat([cell_x.view(-1, 1), cell_y.view(-1, 1)], dim=1)

            wh_restore = (torch.pow(keepbox[:, 2:4] * 2, 2) * keep_anchors) * stride
            xy_restore = (keepbox[:, :2] * 2.0 - 0.5 + keep_cell_xy) * stride
            object_score = layer[keep_batch_indices, keep_anchor_indices, keep_cell_indices, 4]
            object_score = object_score.float().view(-1, 1)
            object_classes = object_classes.float().view(-1 ,1)
            object_classes = torch.zeros_like(object_score)
            keep_batch_indices = keep_batch_indices.float().view(-1, 1)
            box = torch.cat((
                    keep_batch_indices, 
                    xy_restore - (wh_restore - 1) * 0.5, 
                    xy_restore + (wh_restore - 1) * 0.5, 
                    object_score, 
                    object_classes), dim=1)
            objs.append(box)

        if len(objs) > 0:
            objs_cat = torch.cat(objs, dim=0)
            objs_image_base = []
            for ibatch in range(batch):

                select_box = objs_cat[objs_cat[:, 0] == ibatch, 1:]
                objs_image_base.append(select_box)
        else:
            objs_image_base = [torch.zeros((0, 6), device=device) for _ in range(batch)]
        
        if nms_threshold is not None:
            # 使用类内的nms，类间不做操作
            for ibatch in range(batch):
                image_objs = objs_image_base[ibatch]
                if len(image_objs) > 0:
                    max_wh_size = 4096
                    classes = image_objs[:, [5]]
                    bboxes = image_objs[:, :4] + (classes * max_wh_size)
                    confidence = image_objs[:, 4]
                    keep_index = torchvision.ops.boxes.nms(bboxes, confidence, nms_threshold)
                    objs_image_base[ibatch] = image_objs[keep_index]
        return objs_image_base