import torch

class YoloLoss:
    # def __init__(self, anchors, num_classes=80, img_size=(640, 640), device='cpu'):
    def __init__(self, model_info, hyp):
        assert isinstance(model_info, dict), "[LOSS INFO] can not get model info"
        assert isinstance(hyp, dict), "[LOSS INFO] can not get hyp"
        self.model_stride = model_info["model_stride"]
        self.device      = model_info["device"]
        self.anchors     = torch.tensor(
            # 传进来的时候已经是(3, 3, 2), 在这里除以下采样倍数
            model_info["anchors"],
            device=self.device 
        ) / torch.tensor(self.model_stride, device=self.device).view(len(self.model_stride), -1)[:, None]
        self.img_size    = model_info["img_size"]
        self.num_classes = model_info["num_classes"]
        self.num_layers  = model_info["num_layers"]
        self.hyp         = hyp
        # TODO loss相关，后续加入到hyp.yaml中
        self.objloss_layers_ratio = self.hyp.get("objloss_layers_ratio", [0.5, 0.5, 0.5])
        self.auto_objloss_layers_ratio = self.hyp.get("auto_objloss_layers_ratio", True)

    def BCELoss(self, predict, target):
        epsilon = 1e-7
        # 限制最大值与最小值
        predict = (predict >= epsilon).float() * predict + (predict < epsilon).float() * epsilon
        predict = (predict <= (1 - epsilon)).float() * predict + (predict > (1 - epsilon)).float() * (1 - epsilon)
        out = -target * torch.log(predict) - (1 - target) * torch.log(1.0 - predict)
        return out.mean()

    # 入参box必须是最后一维为坐标信息，并且是cx，cy，w，h的形式
    def giou(self, box1, box2):
        # 计算第一个box的相关信息
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2: 4]
        box1_tl = box1_xy - box1_wh / 2.0
        box1_br = box1_xy + box1_wh / 2.0
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]
        # 计算第二个box的相关信息
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2: 4]
        box2_tl = box2_xy - box2_wh / 2.0
        box2_br = box2_xy + box2_wh / 2.0
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]
        # 计算交集并集与IOU
        interSection_tl = torch.max(box1_tl, box2_tl)
        interSection_br = torch.min(box1_br, box2_br)
        interSection_wh = torch.max(interSection_br - interSection_tl, torch.zeros_like(interSection_br))
        interSection_area = interSection_wh[..., 0] * interSection_wh[..., 1]
        union_area = box1_area + box2_area - interSection_area
        iou = interSection_area / union_area
        # 计算最小外接矩形相关信息
        smallest_enclosing_tl = torch.min(box1_tl, box2_tl)
        smallest_enclosing_br = torch.max(box1_br, box2_br)
        smallest_enclosing_wh = torch.max(smallest_enclosing_br - smallest_enclosing_tl, torch.zeros_like(smallest_enclosing_br))
        smallest_enclosing_area = smallest_enclosing_wh[..., 0] * smallest_enclosing_wh[..., 1]

        giou = iou - (smallest_enclosing_area - union_area) / smallest_enclosing_area

        return giou

    def __call__(self, predict, targets):
        loss_cls = torch.zeros(1, device=self.device)
        loss_box = torch.zeros(1, device=self.device)
        loss_obj = torch.zeros(1, device=self.device)
        targets_cls, targets_box, indices, anchors = self.build_targets(predict, targets)

        for i, predict_single_layer in enumerate(predict):
            imi, anch, grid_j, grid_i = indices[i]
            target_obj = torch.zeros(predict_single_layer.shape[:4], dtype=predict_single_layer.dtype, device=self.device)

            num_targets = imi.shape[0]
            if num_targets:
                pxy, pwh, pobj, pcls = predict_single_layer[imi, anch, grid_j, grid_i].split((2, 2, 1, self.num_classes), 1)

                pxy =pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = self.giou(pbox, targets_box[i])
                loss_box += (1.0 - iou).mean()

                # TODO 研究为啥detach脱离计算图了，后续还可以反向传播
                iou = iou.detach().clamp(0).type(target_obj.dtype)
                target_obj[imi, anch, grid_j, grid_i] = iou

                if self.num_classes > 1:
                    target_for_cls = torch.full_like(pcls, 0.0, device=self.device)
                    target_for_cls[range(num_targets), targets_cls[i]] = 1.0
                    loss_cls += self.BCELoss(pcls, target_for_cls)

            loss_obj_single_layer = self.BCELoss(predict_single_layer[..., 4], target_obj)
            # 每个输出层有不同的系数
            loss_obj += loss_obj_single_layer * self.objloss_layers_ratio[i]
            # TODO 后面深入研究一下这样做的好处
            if self.auto_objloss_layers_ratio:
                self.objloss_layers_ratio[i] = self.objloss_layers_ratio[i] * 0.999 + 0.0001 / loss_obj_single_layer.detach().item()
        if self.auto_objloss_layers_ratio:
            # 取下标为1的作为除数，将系数重新变换为以下标为1的系数值为基准的值
            # 如果每次不进行变换，则训练若干轮后，里面的值会很大或很小
            self.objloss_layers_ratio = [x / self.objloss_layers_ratio[1] for x in self.objloss_layers_ratio]
            
        loss_box *= self.hyp['box']
        loss_obj *= self.hyp['obj']
        loss_cls *= self.hyp['cls']
        batch_size = predict[0].shape[0]

        loss_batch = (loss_box + loss_cls + loss_obj) * batch_size
        loss_item = torch.cat((loss_box, loss_cls, loss_obj)).detach()
        return loss_batch, loss_item

        
    def build_targets(self, predict, targets):
        targets_cls, targets_box, indices, anchors_final = [], [], [], []
        num_targets = targets.shape[0]
        num_anchors_single_layer = self.anchors.shape[1] # 3 x numAnchors x 2
        anchors_index_single_layer = torch.arange(num_anchors_single_layer, device=self.device).float().view(num_anchors_single_layer, 1).repeat(1, num_targets)
        gain_single_layer = torch.ones(7, device=self.device)
        # 构建用于单个输出的（yolo有三个输出）targets，三个输出都适用
        targets_single_layer = torch.cat((targets.repeat(num_anchors_single_layer, 1, 1), anchors_index_single_layer[..., None]), 2)

        gain_offset = 0.5
        offset = torch.tensor(
            [
                [0, 0],
                [-1, 0], # 左边扩充
                [1, 0],  # 右边扩充
                [0, -1], # 上边扩充
                [0, 1],  # 下边扩充
            ], device=self.device
        ).float() * gain_offset

        for i in range(self.num_layers):
            anchors, output_layer_shape = self.anchors[i], predict[i].shape
            # 输出的宽和高，用于在输出特征图大小下，还原targets的bbox值，因为targets做了归一化
            gain_single_layer[2:6] = torch.tensor(output_layer_shape)[[3, 2, 3, 2]]

            targets_base_output_layer = targets_single_layer * gain_single_layer
            if num_targets:
                ratio = targets_base_output_layer[..., 4:6] / anchors[:, None]
                # 求真实框和预设锚框满足比例要求的索引：3 x n x 2 -> 3 x n x 2 -> 3 x n
                index = torch.max(ratio, 1 / ratio).max(2)[0] < self.hyp["anchor_t"]
                # 变成二维 m x 7, 此时二维的targets已经包含了三个尺度满足条件的框
                targets_base_output_layer = targets_base_output_layer[index] 
                x_values = targets_base_output_layer[:, 2]
                x_values_inverse = gain_single_layer[2] - x_values
                x_left = (x_values % 1 < gain_offset) & (x_values > 1)
                x_right = (x_values_inverse % 1 < gain_offset) & (x_values_inverse > 1)

                y_values = targets_base_output_layer[:, 3]
                y_values_inverse = gain_single_layer[3] - y_values
                y_top = (y_values % 1 < gain_offset) & (y_values > 1)
                y_bottom = (y_values_inverse % 1 < gain_offset) & (y_values_inverse > 1)

                # 5 x m
                # 注意，这里stack的顺序要和上面的offset的顺序保持一致，
                # 如：offset的顺序是：本身，左边扩充，右边扩充，上边扩充，下边扩充，
                # 则下面stack的时候是(本身，x_left, x_right, y_top, y_bottom)
                # 目的是下面使用index_expand索引取元素并与offset作计算时，能够正确计算
                index_expand = torch.stack((torch.ones_like(x_left), x_left, x_right, y_top, y_bottom))
                offsets = (torch.zeros_like(targets_base_output_layer[:, 2:4])[None] + offset[:, None])[index_expand]
                targets_base_output_layer = targets_base_output_layer.repeat((5, 1, 1))[index_expand]
            
            else:
                targets_base_output_layer = targets_single_layer[0]
                offsets = 0

            imi_clsi, grid_xy, grid_wh, anch = targets_base_output_layer.chunk(4, 1)
            anch, (imi, clsi) = anch.long().view(-1), imi_clsi.long().T
            grid_ij = (grid_xy - offsets).long()
            grid_i, grid_j = grid_ij.T

            # clamp_: inplace方式进行边界控制
            indices.append((imi, anch, grid_j.clamp_(0, output_layer_shape[2] - 1), grid_i.clamp_(0, output_layer_shape[3] - 1)))
            targets_box.append(torch.cat((grid_xy - grid_ij, grid_wh), 1))
            anchors_final.append(anchors[anch])
            targets_cls.append(clsi)

        return targets_cls, targets_box, indices, anchors_final

