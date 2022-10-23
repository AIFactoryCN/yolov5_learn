import os
import sys
import torch

from tqdm import tqdm
from pathlib import Path

from util import *
from metrics import *
from models import heads
from models.yolo import Model
from dataloader import create_dataLoader


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# TODO 重要函数 主要实现真值和预测框的一一对应
# 作用1：对预测框与gt进行一一匹配
# 作用2：对匹配上的预测框进行iou数值判断，用Ture来填充，其余没有匹配上的预测框的所以行数全部设置为False
def process_batch(detections, labels, iou_list):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    # 构建一个[predict_numbers, iouv]全为False的矩阵
    # iou_list = 10 [0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500] (10,)
    correct = torch.zeros(detections.shape[0], iou_list.shape[0], dtype=torch.bool, device=iou_list.device)

    # 计算每个gt与每个predict的iou，shape为: [groundtruth_numbers, predict_numbers]
    iou = box_iou(labels[:, 1:], detections[:, :4])

    # 首先iou >= iou_list[0]：挑选出iou>0.5的所有预测框，进行筛选 [groundtruth_numbers, predict_numbers]
    # 同时labels[:, 0:1] == detections[:, 5]：构建出一个预测类别与真实标签是否相同的矩阵表 [groundtruth_numbers, predict_numbers]
    # 只有同时符合以上两点条件才被赋值为True，此时返回当前矩阵的一个行列索引，x是两个元组 x1, x2
    # x -> ([0, 0, 0], [14, 39, 45])
    # 点(x[0][i], x[1][i])就是符合条件的预测框
    x = torch.where((iou >= iou_list[0]) & (labels[:, 0:1] == detections[:, 5]))

    # 如果存在符合条件的预测框
    if x[0].shape[0]:
        # 将符合条件的位置构建成一个新的矩阵，第一列是行索引（表示gt索引），第二列是列索引（表示预测框索引），第三列是iou值
        # x -> ([0, 0, 0], [14, 39, 45])
        # torch.stack(x, 1)
        # [[ 0, 14],
        #  [ 0, 39],
        #  [ 0, 45]]
        # shape -> num_matched_iou x 3 (3 分别为[label_index, detection_index, iou])
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy() 

        if x[0].shape[0] > 1:
            # argsort获得由小到大排序的索引, [::-1]相当于取反reserve操作，变成由大到小排序的索引
            # 这里是由iou大小对matches矩阵进行排序
            matches = matches[matches[:, 2].argsort()[::-1]]

            # 参数return_index=True：表示会返回唯一值的索引，[0]返回的是唯一值，[1]返回的是索引
            # matches[:, 1]：这里的是获取iou矩阵每个预测框的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
            # 这个操作的含义：每个预测框最多只能出现一次，如果有一个预测框同时和多个gt匹配，只取其最大iou的一个
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            # matches[:, 0]：这里的是获取iou矩阵gt的唯一值，返回的是最大唯一值的索引，因为前面已由大到小排序
            # 这个操作的含义: 每个gt也最多只能出现一次，如果一个gt同时匹配多个预测框，只取其匹配最大的那一个预测框
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        # 以上操作实现了为每一个gt分配一个iou最高的类别的预测框，实现一一对应
        matches = torch.Tensor(matches).to(iou_list.device)

        # 当前获得了gt与预测框的一一对应，其对于的iou可以作为评价指标，构建一个评价矩阵
        # 需要注意，这里的matches[:, 1]表示的是为对应的预测框来赋予其iou所能达到的程度，也就是iou_list的评价指标
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iou_list

    # 在correct中，只有与gt匹配的预测框才有对应的iou评价指标，其他大多数没有匹配的预测框都是全部为False
    return correct


def run(model, 
        test_loader, 
        conf_threshold=0.001,  # confidence threshold
        iou_threshold=0.6,     # NMS IoU threshold
        plots=True,            # 是否将F1，P，PR，R曲线画图并保存
        save_dir=Path(''),
        project=ROOT / 'runs', # save to project
        name='val',            # save to project/name
        exist_ok=False,        # existing project/name ok, do not increment 
        num_classes = 1,
        single_cls = False,    # 是否单类别
        device=None):
    
    # 生成test后评估存储文件夹
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir

    if num_classes == 1:
        single_cls = True

    dt, p, r, f1, mp, mr, map50, map_50_95 = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap_class = [], []
    # start：开始值, end：结束值, steps：分割的点数，默认是100
    # 0.5 -> 0.95指的是iou阈值大小
    # 这里按[0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500]阈值进行iou计算
    iou_list = torch.linspace(0.5, 0.95, 10) # iou vector for mAP@0.5:0.95
    num_iou_threshold = iou_list.numel() # 10
    
    confusion_matrix = ConfusionMatrix(num_classes=num_classes)

    # TODO
    classes_map = ['mouse']
    names = {k: v for k, v in enumerate(classes_map)}

    
    for batch_index, (images, targets, paths, shapes) in enumerate(tqdm(test_loader)):

        device_info = next(model.parameters()).device
        images = images.to(torch.float32)
        images = images.to(device_info)
        images /= 255

        batch_size, _, height, width = images.shape  # batch size, channels, height, width

        # targets.shape [image_index, class_index, cx, cy, width, height]
        targets = targets.to(device_info)

        predicts, predicts_split = model(images)

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device_info)  # to pixels
    
        # agnostic 表示未知，为True时表示单类别，False时为多类别，用于在nms中实现多类别的类内nms
        # nms中有详细注释
        out = non_max_suppression(predicts, conf_threshold, iou_threshold, multi_label=True, agnostic=single_cls)

        # 计算指标过程
        for index, predict in enumerate(out):
            labels = targets[targets[:, 0] == index, 1:]
            nl = len(labels)
            target_class = labels[:, 0].tolist() if nl else []  # target class
            # shapes = (h0, w0), ((h / h0, w / w0), pad)
            shape = shapes[index][0]

            if len(predict) == 0:
                if nl:
                    stats.append((torch.zeros(0, num_iou_threshold, dtype=torch.bool), torch.Tensor(), torch.Tensor(), target_class))
                continue

            # Predictions
            if single_cls:
                predict[:, 5] = 0
            predn = predict.clone()
            # 将预测的坐标信息coords(相对img1_shape)转换回相对原图尺度（img0_shape)
            scale_coords(images[index].shape[1:], predn[:, :4], shape, shapes[index][1])

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                # 将gt box映射回原图大小
                scale_coords(images[index].shape[1:], tbox, shape, shapes[index][1])  # native-space labels
                # 重新构成cls,x,y,x,y格式
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iou_list)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(predict.shape[0], num_iou_threshold, dtype=torch.bool)
            stats.append((correct.cpu(), predict[:, 4].cpu(), predict[:, 5].cpu(), target_class))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # np.array.any()是或操作，任意一个元素为True，输出为True。
    # np.array.all()是与操作，所有元素为True，输出为True。
    if len(stats) and stats[0].any():
        # 根据统计预测结果计算p, r, ap, f1, ap_class（ap_per_class函数是计算每个类的mAP等指标的）
        # p: [nc] 最大平均f1时每个类别的precision
        # r: [nc] 最大平均f1时每个类别的recall
        # ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
        # f1 [nc] 最大平均f1时每个类别的f1
        # ap_class: [nc] 返回数据集中所有的类别index
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap_50_95 = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map_50_95 = p.mean(), r.mean(), ap50.mean(), ap_50_95.mean()
    
    maps = np.zeros(num_classes) + map_50_95
    for i, c in enumerate(ap_class):
        maps[c] = ap_50_95[i]

    return mp, mr, map50, map_50_95, maps


if __name__ == "__main__":
    set_random_seed(3)
    model = Model("yamls/yolov5s.yaml")
    model.eval()
    head = heads.Head()
    test_loader, dataSet = create_dataLoader("/data/data_01/shituo/data/Mouse/mouse/test_list_learn.txt", 640, 2, 32, False)
    model_score = run(model, test_loader)
    print(model_score)
