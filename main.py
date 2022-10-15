import torch
import torch.optim
import os
import numpy as np
import yaml
import math
from copy import deepcopy
from dataloader import createDataLoader
from models.model import DetectionModel
from util import smartOptimizer  
from loss import ComputeLoss           

# --------------------------config-------------------------------
#                       后期放在yaml中读取

device = 'cuda'
# ----model
pretrainedModelPath = ''
modelConfigPath = '/home/bai/bai/sourceCode/baiCode/yolo-rewrite/yamls/yolov5s.yaml'

# -----data
dataPath = '/home/bai/bai/data/A195-20210417(OUT TEST)-train'
imgSizeForNetWork = 640
batchSize = 4
augment = True

# ----optimizer
optimName = 'SGD'
cos_lr = True
epochs = 300

# ----hyp
hypPath = '/home/bai/bai/sourceCode/baiCode/yolo-rewrite/yamls/hyp.yaml'
with open(hypPath, encoding='ascii', errors='ignore') as f:
    hyp = yaml.safe_load(f)

# --------------------------prepare Model-------------------------------
model = DetectionModel(modelConfigPath, inputChannels=3)
exclude = []
if os.path.exists(pretrainedModelPath):
    ckpt = torch.load(pretrainedModelPath, map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    csd = {k: v for k, v in csd.items() if k in model.state_dict() and all(x not in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(csd, strict=False)
model = model.to(device)

# --------------------------prepare Data-------------------------------
trainLoader, dataSet = createDataLoader(dataPath, imgSizeForNetWork, batchSize, augment)
labels = np.concatenate(dataSet.labels, 0)
 
# --------------------------prepare Optimizer Scheduler-------------------------------
optimizer = smartOptimizer(model, optimName, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
if cos_lr:
    lf =lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (hyp['lrf'] - 1) + 1  # cosine 1->hyp['lrf']
else:
    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

# --------------------------prepare Training-------------------------------
# amp = check_amp(model)  # check AMP
model.hyp = hyp
compute_loss = ComputeLoss(model)  # init loss class
scaler = torch.cuda.amp.GradScaler()
nb = len(trainLoader)

# --------------------------Training-------------------------------
for epoch in range(epochs):
    print(f'start epoch {epoch} training')
    model.train()
    mloss = torch.zeros(3, device=device)
    optimizer.zero_grad()
    for i, (imgs, targets, paths, _) in enumerate(trainLoader):
        # ni:一共进行了多少个batch,可以用于warmup
        ni = i + nb * epoch
        imgs = imgs.to(device, non_blocking=True).float() / 255

        with torch.cuda.amp.autocast():
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
    ckpt = {
        'epoch': epoch,
        'best_fitness': 0,
        'model': deepcopy(model).half(),
        'optimizer': optimizer.state_dict(),
    }
    if epoch % 10 == 0:
        torch.save(ckpt, f'./last_{epoch}.pt')