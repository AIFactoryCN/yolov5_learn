import torch.nn as nn
import torch
import math
from copy import deepcopy
from models.common import *
import contextlib

class BaseModel(nn.Module):
    def forward(self, x, profile=False, visualize=False):
        pass


class DetectionModel(nn.Module):
    # def __init__(self, cfg='yolov5s.yaml', inputChannels=3, numClasses=None, anchors=None):
    def __init__(self, cfg='yolov5s.yaml', inputChannels=3, anchors=None):
        super().__init__()
        # 加载yaml配置，可以放在外面加载，只传进来字典
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            import yaml
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.cfg = yaml.safe_load(f)
        inputChannels = self.cfg['inputChannels'] = self.cfg.get('inputChannels', inputChannels)
        # if numClasses and numClasses != self.cfg['numClasses']:
        #     self.cfg['numClasses'] = numClasses
        if anchors:
            self.cfg['anchors'] = round(anchors)

        # 加载网络模型等配置
        self.model, self.save = parseModel(deepcopy(self.cfg), outputChannels=[inputChannels])
        self.names = [str(i) for i in range(self.cfg['numClasses'])]
        self.inplace = self.cfg.get('inplace', True)
        
        # 更新最后一层的属性/信息
        module = self.model[-1]
        if isinstance(module, (Detect)):
            s = 256
            module.inplace = self.inplace
            module.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, inputChannels, s, s))])
            # 更新anchor信息:看大小顺序与stride顺序是否一致
            a = module.anchors.prod(-1).mean(-1).view(-1) # 每个输出的平均锚框大小
            deltaA = a[-1] - a[0]
            deltaS = module.stride[-1] - module.stride[0]
            if deltaA and (deltaA.sign() != deltaS.sign()):
                print('[Anchor Info] Reversing anchor order')
                module.anchors[:] = module.anchors.flip(0)
            module.anchors /= module.stride.view(-1, 1, 1)

            self.stride = module.stride
            self._initialize_biases()

        # 初始化参数
        self._initialize_weights()

    def forward(self, x):
        return self._forward_once(x)

    def _forward_once(self, x):
        y, dt = [], []
        for module in self.model:
            if module.f != -1:
                x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
            x = module(x)
            y.append(x if module.i in self.save else None)
        return x

    def _initialize_biases(self):
        module = self.model[-1]
        for m, s in zip(module.module, module.stride):
            b = m.bias.view(module.numAnchors, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5: 5 + module.numClasses] += math.log(0.6 / (module.numClasses - 0.99999))
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def _initialize_weights(self):
        # 继承nn.Module都有modules()方法
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

class Detect(nn.Module):
    stride = None
    dynamic = False
    export = False

    def __init__(self, numClasses=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.numClasses = numClasses
        self.numOutputs = numClasses + 5
        self.numDetectionLayers = len(anchors)
        self.numAnchors = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.numDetectionLayers)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.numDetectionLayers)]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.numDetectionLayers, -1, 2))
        self.module = nn.ModuleList(nn.Conv2d(x, self.numOutputs * self.numAnchors, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.numDetectionLayers):
            x[i] = self.module[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.numAnchors, self.numOutputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.numClasses + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.numAnchors * nx * ny, self.numOutputs))
        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchros[i].dtype
        shape = 1, self.numAnchors, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.numAnchors, 1, 1, 2)).expand(shape)
        return grid, anchor_grid

def parseModel(modelCfg, outputChannels):
    anchors, numClasses, depth, width, act = modelCfg['anchors'], modelCfg['numClasses'], modelCfg['depth_multiple'], modelCfg['width_multiple'], modelCfg.get('activation')
    numAnchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    numOutputs = numAnchors * (numClasses + 5)

    layers, save, outputChannel = [], [], outputChannels[-1]
    for i, (inputFrom, number, module, args) in enumerate(modelCfg['backbone'] + modelCfg['head']):
        module = eval(module) if isinstance(module, str) else module
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a
        number = max(round(number * depth), 1) if number > 1 else number

        if module in {
            Conv, Bottleneck, SPPF, C3, nn.ConvTranspose2d
            # Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
            # BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x
        }:
            inputChannel, outputChannel = outputChannels[inputFrom], args[0]
            if outputChannel != numOutputs:
                outputChannel = make_divisible(outputChannel * width, 8)
            args = [inputChannel, outputChannel, *args[1:]]
            if module in {C3, '''BottleneckCSP, C3TR, C3Ghost, C3x'''}:
                # 循环次数在内部实现
                args.insert(2, number)
                number = 1
        elif module is nn.BatchNorm2d:
            args = [outputChannels[inputFrom]]
        elif module is Concat:
            outputChannel = sum(outputChannels[x] for x in inputFrom)
        elif module in {Detect, '''Segment'''}:
            args.append([outputChannels[x] for x in inputFrom])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(inputFrom)
            # if module is Segment:
            #     args[3] = make_divisible(args[3] * width, 8)
        elif module is Contract:
            outputChannel = outputChannels[inputFrom] * args[0] ** 2
        elif module is Expand:
            outputChannel = outputChannels[inputFrom] // args[0] ** 2
        else:
            outputChannel = outputChannels[inputFrom]

        module_ = nn.Sequential(*(module(*args) for _ in range(number))) if number > 1 else module(*args)
        t = str(module)[8:-2].replace('__main__.', '')  # module type
        numParams = sum(x.numel() for x in module_.parameters())  # number params
        module_.i, module_.f, module_.type, module_.np = i, inputFrom, t, numParams  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([inputFrom] if isinstance(inputFrom, int) else inputFrom) if x != -1)  # append to savelist
        layers.append(module_)
        if i == 0:
            outputChannels = []
        outputChannels.append(outputChannel)
    return nn.Sequential(*layers), sorted(save)
def make_divisible(x, divisor):
    import math
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor