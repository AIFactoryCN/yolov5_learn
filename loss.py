import torch
import numpy as np
import torch.nn as nn

class YoloLoss(nn.Module):
    # def __init__(self, anchors, num_classes=80, img_size=(640, 640), device='cpu'):
    def __init__(self, model_info, hyp):
        super().__init__()
        assert isinstance(model_info, dict), "[LOSS INFO] can not get model info"
        assert isinstance(hyp, dict), "[LOSS INFO] can not get hyp"
        self.model_stride = model_info["model_stride"]
        self.device      = model_info["device"]
    
        self.num_layers  = model_info["num_layers"]
        anchors = np.array(model_info["anchors"]).reshape(3, self.num_layers, -1)
        self.anchors     = torch.tensor(
            # 传进来的时候已经是(3, 3, 2), 在这里除以下采样倍数
            anchors,
            device=self.device 
        ) / torch.tensor(self.model_stride, device=self.device).view(len(self.model_stride), -1)[:, None]
        self.num_classes = model_info["num_classes"]
        # self.hyp         = hyp
        # # TODO loss相关，后续加入到hyp.yaml中
        # self.objloss_layers_ratio = self.hyp.get("objloss_layers_ratio", [0.5, 0.5, 0.5])
        # self.auto_objloss_layers_ratio = self.hyp.get("auto_objloss_layers_ratio", True)
        
        self.gap_threshold = 4
        # 扩展样本时使用的边界偏移量
        self.offset_boundary = self.anchors.new_tensor([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ])
        self.reduction = "mean"
        self.loss_weight_giou_regression = 0.05
        self.loss_weight_objectness = 1.0
        self.loss_weight_classification = 0.5 * self.num_classes / 80
        self.balance = [4.0, 1.0, 0.4]  # 4, 8, 16 scales loss factor
        self.BCEClassification = nn.BCEWithLogitsLoss(reduction=self.reduction)
        self.BCEObjectness = nn.BCEWithLogitsLoss(reduction=self.reduction)

    # def BCELoss(self, predict, target):
    #     epsilon = 1e-7
    #     # 限制最大值与最小值
    #     predict = (predict >= epsilon).float() * predict + (predict < epsilon).float() * epsilon
    #     predict = (predict <= (1 - epsilon)).float() * predict + (predict > (1 - epsilon)).float() * (1 - epsilon)
    #     out = -target * torch.log(predict) - (1 - target) * torch.log(1.0 - predict)
    #     return out.mean()

    # 入参box必须是最后一维为坐标信息，并且是cx，cy，w，h的形式
    # def giou(self, box1, box2):
    #     # 计算第一个box的相关信息
    #     box1_xy = box1[..., :2]
    #     box1_wh = box1[..., 2: 4]
    #     box1_tl = box1_xy - box1_wh / 2.0
    #     box1_br = box1_xy + box1_wh / 2.0
    #     box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    #     # 计算第二个box的相关信息
    #     box2_xy = box2[..., :2]
    #     box2_wh = box2[..., 2: 4]
    #     box2_tl = box2_xy - box2_wh / 2.0
    #     box2_br = box2_xy + box2_wh / 2.0
    #     box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    #     # 计算交集并集与IOU
    #     interSection_tl = torch.max(box1_tl, box2_tl)
    #     interSection_br = torch.min(box1_br, box2_br)
    #     interSection_wh = torch.max(interSection_br - interSection_tl, torch.zeros_like(interSection_br))
    #     interSection_area = interSection_wh[..., 0] * interSection_wh[..., 1]
    #     union_area = box1_area + box2_area - interSection_area
    #     iou = interSection_area / union_area
    #     # 计算最小外接矩形相关信息
    #     smallest_enclosing_tl = torch.min(box1_tl, box2_tl)
    #     smallest_enclosing_br = torch.max(box1_br, box2_br)
    #     smallest_enclosing_wh = torch.max(smallest_enclosing_br - smallest_enclosing_tl, torch.zeros_like(smallest_enclosing_br))
    #     smallest_enclosing_area = smallest_enclosing_wh[..., 0] * smallest_enclosing_wh[..., 1]

    #     giou = iou - (smallest_enclosing_area - union_area) / smallest_enclosing_area

    #     return giou
    def giou(self, a, b):
        '''
        计算a与b的GIoU
        参数：
        a[Nx4]：      要求是[cx, cy, width, height]
        b[Nx4]:       要求是[cx, cy, width, height]
        '''
        # a is n x 4
        # b is n x 4

        # cx, cy, width, height
        a_xmin, a_xmax = a[:, 0] - a[:, 2] / 2, a[:, 0] + a[:, 2] / 2
        a_ymin, a_ymax = a[:, 1] - a[:, 3] / 2, a[:, 1] + a[:, 3] / 2
        b_xmin, b_xmax = b[:, 0] - b[:, 2] / 2, b[:, 0] + b[:, 2] / 2
        b_ymin, b_ymax = b[:, 1] - b[:, 3] / 2, b[:, 1] + b[:, 3] / 2

        inter_xmin = torch.max(a_xmin, b_xmin)
        inter_xmax = torch.min(a_xmax, b_xmax)
        inter_ymin = torch.max(a_ymin, b_ymin)
        inter_ymax = torch.min(a_ymax, b_ymax)
        inter_width = (inter_xmax - inter_xmin).clamp(0)
        inter_height = (inter_ymax - inter_ymin).clamp(0)
        inter_area = inter_width * inter_height

        a_width, a_height = (a_xmax - a_xmin), (a_ymax - a_ymin)
        b_width, b_height = (b_xmax - b_xmin), (b_ymax - b_ymin)
        union = (a_width * a_height) + (b_width * b_height) - inter_area
        iou = inter_area / union

        # smallest enclosing box
        convex_width = torch.max(a_xmax, b_xmax) - torch.min(a_xmin, b_xmin) + 1e-16
        convex_height = torch.max(a_ymax, b_ymax) - torch.min(a_ymin, b_ymin)
        convex_area = convex_width * convex_height + 1e-16
        return iou - (convex_area - union) / convex_area

    def forward(self, predict, targets):
        # bbox is [image_id, classes_id, cx, cy, width, height]
        # targets[num_targets, bbox]
        device = targets.device
        loss_classification = torch.FloatTensor([0]).to(device)
        loss_box_regression = torch.FloatTensor([0]).to(device)
        loss_objectness = torch.FloatTensor([0]).to(device)
        num_target = targets.shape[0]

        for ilayer, layer in enumerate(predict):
            layer_height, layer_width = layer.shape[-2:]

            # batch_size, num_anchors, height, width, 6[x, y, r, b, objectness, classess]
            layer = layer.view(-1, 3, 5 + self.num_classes, layer_height, layer_width).permute(0, 1, 3, 4, 2).contiguous()

            # image_id, classes_id, cx, cy, width, height
            # targets is NumTarget x 6
            layer_anchors = self.anchors[ilayer]
            num_anchor = layer_anchors.shape[0]
            feature_size_gain = targets.new_tensor([1, 1, layer_width, layer_height, layer_width, layer_height])
            targets_feature_scale = targets * feature_size_gain
            anchors_wh = layer_anchors.view(num_anchor, 1, 2)
            targets_wh = targets_feature_scale[:, 4:6].view(1, num_target, 2)

            # # wh_ratio is [num_anchor, num_target, 2]
            wh_ratio = targets_wh / anchors_wh

            # # select_mask is [num_anchor, num_target]
            max_wh_ratio_values, _ = torch.max(wh_ratio, 1 / wh_ratio).max(2)
            select_mask = max_wh_ratio_values < self.gap_threshold

            # NumTarget x num_anchor, 1
            # target -> anchor
            # targets.repeat(num_anchor, 1, 1) -> num_anchor x NumTarget x 6
            # select_targets is [matched_num_target, 6]
            select_targets = targets_feature_scale.repeat(num_anchor, 1, 1)[select_mask]
            matched_num_target = len(select_targets)

            featuremap_objectness = layer[..., 4]
            objectness_ground_truth = torch.zeros_like(featuremap_objectness, device=device)

            if matched_num_target > 0:
                #  0  0  0  0  0  0  0  ...   num_target
                #  1  1  1  1  1  1  1 
                #  2  2  2  2  2  2  2
                # anchor_index_repeat is [num_anchor, num_target]
                anchor_index_repeat = torch.arange(num_anchor, device=device).view(num_anchor, 1).repeat(1, num_target)

                # select_anchor_index is [matched_num_target]
                select_anchor_index = anchor_index_repeat[select_mask]

                # 扩展采样，在原本中心的位置上增加采样点，根据中心坐标，x、y距离谁近，选择一个谁
                # 这里不考虑cx, cy正好为0.5的情况，则等于将样本增加2倍
                # select_targets_xy is [matched_num_target, 2]
                select_targets_xy = select_targets[:, 2:4]
                xy_divided_one_remainder = select_targets_xy % 1.0
                coord_cell_middle = 0.5
                feature_map_low_boundary = 1.0
                feature_map_high_boundary = feature_size_gain[[2, 3]] - 1.0
                less_x_matched, less_y_matched = ((xy_divided_one_remainder < coord_cell_middle) & (select_targets_xy > feature_map_low_boundary)).t()
                greater_x_matched, greater_y_matched = ((xy_divided_one_remainder > (1 - coord_cell_middle)) & (select_targets_xy < feature_map_high_boundary)).t()

                select_anchor_index = torch.cat([
                    select_anchor_index, 
                    select_anchor_index[less_x_matched],
                    select_anchor_index[less_y_matched],
                    select_anchor_index[greater_x_matched],
                    select_anchor_index[greater_y_matched]
                ], dim=0)

                select_targets = torch.cat([
                    select_targets, 
                    select_targets[less_x_matched],
                    select_targets[less_y_matched],
                    select_targets[greater_x_matched],
                    select_targets[greater_y_matched]
                ], dim=0)

                xy_offsets = torch.zeros_like(select_targets_xy, device=device)
                xy_offsets = torch.cat([
                    xy_offsets, 
                    xy_offsets[less_x_matched] + self.offset_boundary[0],
                    xy_offsets[less_y_matched] + self.offset_boundary[1],
                    xy_offsets[greater_x_matched] + self.offset_boundary[2],
                    xy_offsets[greater_y_matched] + self.offset_boundary[3]
                ]) * coord_cell_middle

                # image_id, classes_id, cx, cy, width, height
                # .t()的目的是把nx2转置为2xn，可以直接解包为2个变量，一个变量为一行
                matched_extend_num_target = select_targets.shape[0]
                gt_image_id, gt_classes_id = select_targets[:, :2].long().t()
                gt_xy = select_targets[:, 2:4]
                gt_wh = select_targets[:, 4:6]
                grid_xy = (gt_xy - xy_offsets).long()
                grid_x, grid_y = grid_xy.t()
                gt_xy = gt_xy - grid_xy

                # select_anchors is [matched_extend_num_target, 2]
                select_anchors = layer_anchors[select_anchor_index]

                #####################################################
                # object_position_predict is [matched_extend_num_target, 6]
                object_position_predict = layer[gt_image_id, select_anchor_index, grid_y, grid_x]
                object_predict_xy = (object_position_predict[:, :2].sigmoid() * 2.0 - 0.5).float()
                object_predict_wh = torch.pow(object_position_predict[:, 2:4].sigmoid() * 2.0,  2.0) * select_anchors

                # matched_extend_num_target, 4
                object_predict_box = torch.cat((object_predict_xy, object_predict_wh), dim=1)
                object_ground_truth_box = torch.cat((gt_xy, gt_wh), dim=1)
                gious = self.giou(object_predict_box, object_ground_truth_box)
                giou_loss = 1.0 - gious
                loss_box_regression += giou_loss.mean() if self.reduction == "mean" else giou_loss.sum()
                objectness_ground_truth[gt_image_id, select_anchor_index, grid_y, grid_x] = gious.detach().clamp(0).type(objectness_ground_truth.dtype)

                if self.num_classes > 1:
                    # classification loss (only if multiple classes)
                    # object_classification is [matched_extend_num_target, num_classes]
                    object_classification = object_position_predict[:, 5:]
                    classification_targets = torch.full_like(object_classification, 0, device=device)
                    classification_targets[torch.arange(matched_extend_num_target), gt_classes_id] = 1.0
                    loss_classification += self.BCEClassification(object_classification, classification_targets)

            # batch, num_anchors, height, width, 6[x, y, r, b, objectness, classess]
            # batch, num_anchors, height, width
            loss_objectness += self.BCEObjectness(featuremap_objectness, objectness_ground_truth) * self.balance[ilayer]

        num_predict = len(predict)
        scale = 3 / num_predict
        batch_size = predict[0].shape[0]
        loss_box_regression *= self.loss_weight_giou_regression * scale
        loss_objectness *= self.loss_weight_objectness * scale * (1.4 if num_predict == 4 else 1.0)
        loss_classification *= self.loss_weight_classification * scale
        
        loss = loss_box_regression + loss_objectness + loss_classification
        loss_visual = f"Loss: {loss.item():.06f}, Box: {loss_box_regression.item():.06f}, Obj: {loss_objectness.item():.06f}, Cls: {loss_classification.item():.06f}"
        return loss * batch_size, loss_visual

    # def __call__(self, predict, targets):
    #     loss_cls = torch.zeros(1, device=self.device)
    #     loss_box = torch.zeros(1, device=self.device)
    #     loss_obj = torch.zeros(1, device=self.device)
    #     targets_cls, targets_box, indices, anchors = self.build_targets(predict, targets)

    #     for i, predict_single_layer in enumerate(predict):
    #         # n, anchor_add_class, fea_h, fea_w = predict_single_layer.shape
    #         # predict_single_layer = predict_single_layer.view(n, 3, self.num_classes + 5, fea_h, fea_w)
    #         # predict_single_layer = predict_single_layer.permute(0, 1, 3, 4, 2).contiguous()

    #         # torch.Size([2, 75, 80, 80])
    #         imi, anch, grid_j, grid_i = indices[i]
    #         target_obj = torch.zeros(predict_single_layer.shape[:4], dtype=predict_single_layer.dtype, device=self.device)

    #         # print(predict_single_layer[imi, anch, grid_j, grid_i].shape)
    #         num_targets = imi.shape[0]
    #         if num_targets:
    #             pxy, pwh, pobj, pcls = predict_single_layer[imi, anch, grid_j, grid_i].split((2, 2, 1, self.num_classes), 1)

    #             pxy  = pxy.sigmoid() * 2 - 0.5
    #             pwh  = (pwh.sigmoid() * 2) ** 2 * anchors[i]
    #             pbox = torch.cat((pxy, pwh), 1)
    #             iou  = self.giou(pbox, targets_box[i])
    #             loss_box += (1.0 - iou).mean()

    #             # TODO 研究为啥detach脱离计算图了，后续还可以反向传播
    #             iou = iou.detach().clamp(0).type(target_obj.dtype)
    #             target_obj[imi, anch, grid_j, grid_i] = iou

    #             if self.num_classes > 1:
    #                 target_for_cls = torch.full_like(pcls, 0.0, device=self.device)
    #                 target_for_cls[range(num_targets), targets_cls[i]] = 1.0
    #                 loss_cls += self.BCELoss(pcls, target_for_cls)

    #         loss_obj_single_layer = self.BCELoss(predict_single_layer[..., 4], target_obj)
    #         # 每个输出层有不同的系数
    #         loss_obj += loss_obj_single_layer * self.objloss_layers_ratio[i]
    #         # TODO 后面深入研究一下这样做的好处
    #         if self.auto_objloss_layers_ratio:
    #             self.objloss_layers_ratio[i] = self.objloss_layers_ratio[i] * 0.999 + 0.0001 / loss_obj_single_layer.detach().item()
    #     if self.auto_objloss_layers_ratio:
    #         # 取下标为1的作为除数，将系数重新变换为以下标为1的系数值为基准的值
    #         # 如果每次不进行变换，则训练若干轮后，里面的值会很大或很小
    #         self.objloss_layers_ratio = [x / self.objloss_layers_ratio[1] for x in self.objloss_layers_ratio]
            
    #     loss_box *= self.hyp['box']
    #     loss_obj *= self.hyp['obj']
    #     loss_cls *= self.hyp['cls']
    #     batch_size = predict[0].shape[0]

    #     loss_batch = (loss_box + loss_cls + loss_obj) * batch_size
    #     loss_item = torch.cat((loss_box, loss_cls, loss_obj)).detach()
    #     return loss_batch, loss_item

        
    # def build_targets(self, predict, targets):
    #     targets_cls, targets_box, indices, anchors_final = [], [], [], []
    #     num_targets = targets.shape[0]
    #     num_anchors_single_layer = self.anchors.shape[1] # 3 x numAnchors x 2
    #     anchors_index_single_layer = torch.arange(num_anchors_single_layer, device=self.device).float().view(num_anchors_single_layer, 1).repeat(1, num_targets)
        
    #     gain_single_layer = torch.ones(7, device=self.device)
    #     # 构建用于单个输出的（yolo有三个输出）targets，三个输出都适用
    #     targets_single_layer = torch.cat((targets.repeat(num_anchors_single_layer, 1, 1), anchors_index_single_layer[..., None]), 2)

    #     gain_offset = 0.5
    #     offset = torch.tensor(
    #         [
    #             [0, 0],
    #             [-1, 0], # 左边扩充
    #             [1, 0],  # 右边扩充
    #             [0, -1], # 上边扩充
    #             [0, 1],  # 下边扩充
    #         ], device=self.device
    #     ).float() * gain_offset

    #     for i in range(self.num_layers):
    #         anchors, output_layer_shape = self.anchors[i], predict[i].shape
    #         # 输出的宽和高，用于在输出特征图大小下，还原targets的bbox值，因为targets做了归一化
    #         gain_single_layer[2:6] = torch.tensor(output_layer_shape)[[3, 2, 3, 2]]

    #         targets_base_output_layer = targets_single_layer * gain_single_layer
    #         if num_targets:
    #             ratio = targets_base_output_layer[..., 4:6] / anchors[:, None]
    #             # 求真实框和预设锚框满足比例要求的索引：3 x n x 2 -> 3 x n x 2 -> 3 x n
    #             index = torch.max(ratio, 1 / ratio).max(2)[0] < self.hyp["anchor_t"]
    #             # 变成二维 m x 7, 此时二维的targets已经包含了三个尺度满足条件的框
    #             targets_base_output_layer = targets_base_output_layer[index] 
    #             x_values = targets_base_output_layer[:, 2]
    #             x_values_inverse = gain_single_layer[2] - x_values
    #             x_left = (x_values % 1 < gain_offset) & (x_values > 1)
    #             x_right = (x_values_inverse % 1 < gain_offset) & (x_values_inverse > 1)

    #             y_values = targets_base_output_layer[:, 3]
    #             y_values_inverse = gain_single_layer[3] - y_values
    #             y_top = (y_values % 1 < gain_offset) & (y_values > 1)
    #             y_bottom = (y_values_inverse % 1 < gain_offset) & (y_values_inverse > 1)

    #             # 5 x m
    #             # 注意，这里stack的顺序要和上面的offset的顺序保持一致，
    #             # 如：offset的顺序是：本身，左边扩充，右边扩充，上边扩充，下边扩充，
    #             # 则下面stack的时候是(本身，x_left, x_right, y_top, y_bottom)
    #             # 目的是下面使用index_expand索引取元素并与offset作计算时，能够正确计算
    #             index_expand = torch.stack((torch.ones_like(x_left), x_left, x_right, y_top, y_bottom))
    #             offsets = (torch.zeros_like(targets_base_output_layer[:, 2:4])[None] + offset[:, None])[index_expand]
    #             targets_base_output_layer = targets_base_output_layer.repeat((5, 1, 1))[index_expand]
            
    #         else:
    #             targets_base_output_layer = targets_single_layer[0]
    #             offsets = 0

    #         imi_clsi, grid_xy, grid_wh, anch = targets_base_output_layer.chunk(4, 1)
    #         anch, (imi, clsi) = anch.long().view(-1), imi_clsi.long().T
    #         grid_ij = (grid_xy - offsets).long()
    #         grid_i, grid_j = grid_ij.T

    #         # clamp_: inplace方式进行边界控制
    #         indices.append((imi, anch, grid_j.clamp_(0, output_layer_shape[2] - 1), grid_i.clamp_(0, output_layer_shape[3] - 1)))
    #         targets_box.append(torch.cat((grid_xy - grid_ij, grid_wh), 1))
    #         anchors_final.append(anchors[anch])
    #         targets_cls.append(clsi)

    #     return targets_cls, targets_box, indices, anchors_final

