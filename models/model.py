import torch.nn as nn
import torch
import math
from copy import deepcopy
from models.common import *
import contextlib



def make_divisible(x, divisor):
    import math
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', input_channels=3, anchors=None):
        super().__init__()
        # 加载yaml配置，可以放在外面加载，只传进来字典
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            import yaml
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.cfg = yaml.safe_load(f)
        input_channels = self.cfg['input_channels'] = self.cfg.get('input_channels', input_channels)
  
        if anchors:
            self.cfg['anchors'] = round(anchors)

        # 加载网络模型等配置
        self.model, self.saved_index = parse_model(deepcopy(self.cfg), all_output_channels=[input_channels])
        self.names = [str(i) for i in range(self.cfg['num_classes'])]
        self.inplace = self.cfg.get('inplace', True)
        
        # 更新最后一层的属性/信息
        module = self.model[-1]
        if isinstance(module, (Detect)):
            s = 256
            module.inplace = self.inplace
            module.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, input_channels, s, s))])
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
        
        y = []
        for module in self.model:
            if module.input_from != -1:
                if isinstance(module.input_from, int):
                    x = y[module.input_from]
                else:
                    xout = []
                    for i in module.input_from:
                        if i == -1:
                            xval = x
                        else:
                            xval = y[i]
                        xout.append(xval)
                    x = xout
            
            x = module(x)
            y.append(x if module.layer_index in self.saved_index else None)
        return x

        # y, dt = [], []
        # for module in self.model:
        #     if module.f != -1:
        #         x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
        #     x = module(x)
        #     y.append(x if module.i in self.save else None)
        # return x

    def _initialize_biases(self):
        module = self.model[-1]
        for m, s in zip(module.module, module.stride):
            b = m.bias.view(module.num_anchors, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5: 5 + module.num_classes] += math.log(0.6 / (module.num_classes - 0.99999))
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
    
    # TODO conv and bn fuse


class Detect(nn.Module):
    stride = None
    dynamic = False
    export = False

    def __init__(self, num_classes=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.num_detection_layers = len(anchors)
        self.num_anchors = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.num_detection_layers)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.num_detection_layers)]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.num_detection_layers, -1, 2))
        self.module = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []
        for i in range(self.num_detection_layers):
            x[i] = self.module[i](x[i])
            batch_size, _, layer_height, layer_width = x[i].shape
            x[i] = x[i].view(batch_size, self.num_anchors, self.num_outputs, layer_height, layer_width).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(layer_width, layer_height, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.num_classes + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(batch_size, self.num_anchors * layer_width * layer_height, self.num_outputs))
        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchros[i].dtype
        shape = 1, self.num_anchors, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.num_anchors, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


def parse_model(model_config, all_output_channels):
    anchors, num_classes, depth, width= model_config['anchors'], model_config['num_classes'], model_config['depth_multiple'], model_config['width_multiple']
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    num_outputs = num_anchors * (num_classes + 5)

    layers, save, output_channel = [], [], all_output_channels[-1]
    for layer_index, (input_from, repeat, module, args) in enumerate(model_config['backbone'] + model_config['head']):
        module = eval(module) if isinstance(module, str) else module

        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a

        repeat = max(round(repeat * depth), 1) if repeat > 1 else repeat

        if module in [
            Conv, Bottleneck, SPPF, C3, BottleneckCSP, nn.ConvTranspose2d]:

            input_channel, output_channel = all_output_channels[input_from], args[0]
            if output_channel != num_outputs:
                output_channel = make_divisible(output_channel * width, 8)
            args = [input_channel, output_channel, *args[1:]]
            if module in [C3, BottleneckCSP]:
                # 循环次数在内部实现
                args.insert(2, repeat)
                repeat = 1
        elif module is nn.BatchNorm2d:
            args = [all_output_channels[input_from]]
        elif module is Concat:
            output_channel = sum(all_output_channels[x] for x in input_from)
        elif module is Detect:
            input_channel = [all_output_channels[x] for x in input_from]
            args = [num_classes, num_anchors, input_channel]
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(input_from)
        elif module is Contract:
            output_channel = all_output_channels[input_from] * args[0] ** 2
        elif module is Expand:
            output_channel = all_output_channels[input_from] // args[0] ** 2
        else:
            output_channel = all_output_channels[input_from]

        module_ = nn.Sequential(*(module(*args) for _ in range(repeat))) if repeat > 1 else module(*args)
        module_type = str(module)[8:-2].replace('__main__.', '')  # module type
        num_params = sum(x.numel() for x in module_.parameters())  # number params
        module_.layer_index, module_.input_from, module_.type, module_.num_params = layer_index, input_from, module_type, num_params  # attach index, 'from' index, type, number params
        # save.extend(x % layer_index for x in ([input_from] if isinstance(input_from, int) else input_from) if x != -1)  # append to savelist
        if not isinstance(input_from, list):
                input_from = [input_from]
        
        # save 保存的是搭建模型时在neck部分仍需用的的model layer
        save.extend(filter(lambda x: x!=-1, input_from))
        layers.append(module_)

        if layer_index == 0:
            all_output_channels = []
        all_output_channels.append(output_channel)

    return nn.Sequential(*layers), sorted(save)




if __name__ == "__main__":

    model = Model(cfg="yamls/yolov5s.yaml")
    input = torch.zeros((1, 3, 640, 640))
    predict = model(input)
    print(predict[0].shape)