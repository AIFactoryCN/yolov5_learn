import torch
import torch.optim
import os
import numpy as np
import yaml
import math
import argparse
from copy import deepcopy
from dataloader import createDataLoader
from models.model import Model
from util import use_optimizer  
from loss import YoloLoss        


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


def main(opt):

    # --------------------------读取配置-------------------------------
    model_info      = {}
    hyp_path        = opt.hyp
    config_file     = opt.cfg
    image_size      = opt.img_size
    epochs          = opt.epochs
    device          = opt.device
    pretrained_path = opt.pretrained_path
    data_path       = opt.data
    batch_size      = opt.batch_size
    augment = True
    # ----optimizer
    optimName = 'SGD'
    cos_lr = True
    
    with open(hyp_path, encoding='ascii', errors='ignore') as f:
        hyp = yaml.safe_load(f)

    # --------------------------数据加载及锚框自动聚类-------------------------------
    train_dataloader, dataSet = createDataLoader(data_path, image_size, batch_size, max_stride=32, augment=True)
    labels = np.concatenate(dataSet.labels, 0)
    # TODO 锚框自动聚类

    # --------------------------更新整体信息到一个字典中-------------------------------
    # TODO 先临时写死，后期优化
    model_info = {
        "anchors":[
            [[10, 13], [16, 30], [33, 23]], 
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ],
        # img_size的用处：
        # 1. 将数据缩放成 img_size大小的正方形（等比缩放短边用常数填充）
        # 2. 构建网络时，判断是否能整除网络的最大缩放倍数
        "img_size": (640, 640),
        "num_classes": 20,
        "num_layers": 3,
        "device": "cuda",
        # TODO 先暂时写死，方便调试，后续根据实际的输入输出计算
        "model_stride": [8, 16, 32],
    }
 

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
    
    ema = ModelEMA(model)

    best_model_score = 0.0
    momentum = 0.937
    weight_decay = 5e-4
    basic_number_of_batch_size = 64
    accumulate = max(round(basic_number_of_batch_size / batch_size), 1)  # accumulate loss before optimizing
    #todo: 权重衰减
    weight_decay *= batch_size * accumulate / basic_number_of_batch_size 


    # --------------------------准备优化器、学习策略等-------------------------------
    optimizer = use_optimizer(model, optimName, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    if cos_lr:
        lf =lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (hyp['lrf'] - 1) + 1  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # --------------------------准备损失函数等-------------------------------
    # amp = check_amp(model)  # check AMP
    model.hyp = hyp
    compute_loss = YoloLoss(model_info, hyp)  # init loss class
    # scaler = torch.cuda.amp.GradScaler()
    num_iter_per_epoch = len(train_dataloader)

    last_opt_step = -1

    nw = max(3 * num_iter_per_epoch, 1e3)

    # --------------------------开始训练-------------------------------
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
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0

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
                if ema is not None:
                    ema.update(model)

            current_epoch = epoch + (i + 1) / num_iter_per_epoch
            if num_iter % 20 == 0:
                print(f"accumulate: {accumulate}")
                log_line = f"Epoch: {current_epoch:.2f}/{epochs}, Iter: {i}, Targets: {num_targets}, LR: {learning_rate:.5f}, {loss_items}"
                print(log_line)

        scheduler.step()

        # TODO test
        test_dataloader = train_dataloader

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
    parser.add_argument('--data', type=str, default='../datasets/VOC', help='数据地址')
    parser.add_argument('--pretrained_path', type=str, default='', help='预训练模型')
    parser.add_argument('--hyp', type=str, default='yamls/hyp.yaml', help='训练超参数')
    parser.add_argument('--img_size', type=int, default=640, help='图片输入尺寸')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--epochs', type=int, default=10, help='训练总轮数')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
