import torch.nn as nn
import torch

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=1, stride=1, padding=None, groups=1, dilation=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, autopad(kernel_size, padding, dilation), groups=groups, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.act = nn.SiLU() if activation is True else activation if isinstance(activation, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, input_channel, output_channel, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        middle_channel = int(output_channel * expansion)
        self.cv1 = Conv(input_channel, middle_channel, 1, 1)
        self.cv2 = Conv(middle_channel, output_channel, 3, 1, groups=groups)
        self.add = shortcut and middle_channel == output_channel
    
    def forward(self, x):
        # 这里能直接 + 的原因是，Conv模块里面有自动pad，输出尺寸不变
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    
    def __init__(self, input_channel, output_channel, repeat=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        middle_channel = int(output_channel * expansion) 
        self.cv1 = Conv(input_channel, middle_channel, 1, 1)
        self.cv2 = nn.Conv2d(input_channel, middle_channel, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(middle_channel, middle_channel, 1, 1, bias=False)
        self.cv4 = Conv(2 * middle_channel, output_channel, 1, 1)
        self.bn = nn.BatchNorm2d(2 * middle_channel)  # cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(middle_channel, middle_channel, shortcut, groups, expansion=1.0) for _ in range(repeat)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    def __init__(self, input_channel, output_channel, repeat=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        middle_channel = int(output_channel * expansion)
        self.cv1 = Conv(input_channel, middle_channel, 1, 1)
        self.cv2 = Conv(input_channel, middle_channel, 1, 1)
        self.cv3 = Conv(2 * middle_channel, output_channel, 1)
        self.m = nn.Sequential(*(Bottleneck(middle_channel, middle_channel, shortcut, groups, expansion=1.0) for _ in range(repeat)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer
    def __init__(self, input_channel, output_channel, kernel_size_list=(5, 9, 13)):
        super(SPP, self).__init__()
        middle_channel = input_channel // 2
        self.cv1 = Conv(input_channel, middle_channel, 1, 1)
        self.cv2 = Conv(middle_channel * (len(kernel_size_list) + 1), output_channel, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2) for kernel_size in kernel_size_list])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class SPPF(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=5):
        super().__init__()
        middle_channel = input_channel // 2
        self.cv1 = Conv(input_channel, middle_channel, 1, 1)
        self.cv2 = Conv(middle_channel * 4, output_channel, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, dim=self.dimension)


class Contract():
    pass


class Expand():
    pass