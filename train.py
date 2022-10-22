import torch
import torch.optim
import os
import numpy as np
import yaml
import math
from copy import deepcopy
import argparse
from dataloader import create_dataLoader
from models.model import Model
from util import use_optimizer, ModelEMA  
from loss import YoloLoss     
import test   



def main(opt):
    # --------------------------读取配置-------------------------------
    hyp_path        = opt.hyp
    config_file     = opt.cfg
    image_size      = opt.img_size
    epochs          = opt.epochs
    device          = opt.device
    pretrained_path = opt.pretrained_path
    train_data_path = opt.data
    test_data_path  = opt.test_data
    batch_size      = opt.batch_size
    augment         = opt.augment
    ema             = opt.ema

    # ----optimizer
    optimName = 'SGD'
    cos_lr = True
    
    with open(hyp_path, encoding='ascii', errors='ignore') as f:
        hyp = yaml.safe_load(f)

    # --------------------------数据加载及锚框自动聚类-------------------------------
    train_dataloader, dataSet = create_dataLoader(train_data_path, image_size, batch_size, max_stride=32, augment=augment)
    test_dataloader, _ = create_dataLoader(test_data_path, image_size, batch_size, max_stride=32, augment=augment)
    
    # TODO 锚框自动聚类

    # --------------------------更新整体信息到一个字典中-------------------------------
    model_info      = {}
    with open(config_file, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)
    anchors = cfg['anchors']
    num_classes = cfg['num_classes']
    num_layers = np.array(anchors).shape[1] // 2
    model_info['anchors'] = anchors
    model_info['num_classes'] = num_classes
    model_info['num_layers'] = num_layers
    model_info['model_stride'] = [8, 16, 32]
    model_info['device'] = device

    # --------------------------准备网络模型-------------------------------
    model = Model(config_file, input_channels=3)
    exclude = []
    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        csd = ckpt['model'].float().state_dict()
        csd = {k: v for k, v in csd.items() if k in model.state_dict() and all(x not in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        model.load_state_dict(csd, strict=False)
    model = model.to(device)

    # ------------------------- 指数移动平均 ---------------------------------------
    if ema:
        ema = ModelEMA(model)

    # --------------------------准备优化器、学习策略等-------------------------------
    weight_decay = 5e-4
    basic_number_of_batch_size = 64
    accumulate = max(round(basic_number_of_batch_size / batch_size), 1)  # accumulate loss before optimizing
    #todo: 权重衰减
    weight_decay *= batch_size * accumulate / basic_number_of_batch_size 

    optimizer = use_optimizer(model, optimName, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    if cos_lr:
        lf =lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (hyp['lrf'] - 1) + 1  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # --------------------------准备损失函数等-------------------------------
    compute_loss = YoloLoss(model_info, hyp)  # init loss class
    num_iter_per_epoch = len(train_dataloader)
    
    # --------------------------开始训练-------------------------------
    nw = max(3 * num_iter_per_epoch, 1e3)
    for epoch in range(epochs):
        print(f'start epoch {epoch} training')
        model.train()

        learning_rate = scheduler.get_last_lr()[0]
        optimizer.zero_grad()
        for i, (images, targets, paths, _) in enumerate(train_dataloader):
            # num_iter:一共进行了多少个batch,可以用于warmup
            num_iter = i + num_iter_per_epoch * epoch
            num_targets = targets.shape[0]

            if num_iter <= nw:
                xi = [0, nw]  # x interp
                for j, x in enumerate(optimizer.param_groups):
                    accumulate = max(1, np.interp(num_iter, xi, [1, basic_number_of_batch_size / batch_size]).round())

                    bias_param_group_index = 2
                    x['lr'] = np.interp(num_iter, xi, [0.1 if j == bias_param_group_index else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iter, xi, [0.9, hyp['momentum']])

            images = images.to(device, non_blocking=True).float() / 255

            with torch.cuda.amp.autocast():
                predict = model(images)
                loss, loss_items = compute_loss(predict, targets.to(device))

            loss.backward()

            #todo: 梯度累计叠加好几次, 再去更新
            #使得batchsize 隐性变大
            if num_iter % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            current_epoch = epoch + (i + 1) / num_iter_per_epoch
            if num_iter % 20 == 0:
                print(f"accumulate: {accumulate}")
                log_line = f"Epoch: {current_epoch:.2f}/{epochs}, Iter: {i}, Targets: {num_targets}, LR: {learning_rate:.5f}, {loss_items}"
                print(log_line)

        scheduler.step()

        # TODO test
        test_dataloader = test_dataloader
        model.eval()
        model_score = test.run(model, test_dataloader)
        print(model_score)

        # score = test.run(model, test_dataloader)

        ckpt = {
            'epoch': epoch,
            'best_fitness': 0,
            'model': deepcopy(model).half(),
            'optimizer': optimizer.state_dict(),
        }
        if epoch % 10 == 0:
            torch.save(ckpt, f'./last_{epoch}.pt')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yamls/yolov5s.yaml', help='模型配置')
    parser.add_argument('--data', type=str, default='/data/data_01/shituo/data/Mouse/mouse/test_list_learn.txt', help='训练数据地址')
    parser.add_argument('--test_data', type=str, default='/data/data_01/shituo/data/Mouse/mouse/test_list_learn.txt', help='测试数据地址')
    parser.add_argument('--pretrained_path', type=str, default='', help='预训练模型')
    parser.add_argument('--hyp', type=str, default='yamls/hyp.yaml', help='训练超参数')
    parser.add_argument('--img_size', type=int, default=640, help='图片输入尺寸')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备')
    parser.add_argument('--epochs', type=int, default=10, help='训练总轮数')
    parser.add_argument('--augment', type=bool, default=True, help='使用数据增强')
    parser.add_argument('--ema', type=bool, default=False, help='使用指数移动平均')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
