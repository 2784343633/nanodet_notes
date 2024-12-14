> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/NeoZng/article/details/123299682)

 _**--neozng1@hnu.edu.cn**_

> 笔者已经为 [nanodet](https://so.csdn.net/so/search?q=nanodet&spm=1001.2101.3001.7020) 增加了非常详细的注释，代码请戳此仓库：[nanodet_detail_notes: detail every detail about nanodet](https://gitee.com/neozng1/nanodet_detail_notes "nanodet_detail_notes: detail every detail about nanodet") 。
> 
> 此仓库会跟着文章推送的节奏持续更新！

**目录**

[1. Backbone](#1.%20Backbone)

[1.0. _make_divisible()](#t0)

[1.1. SqueezeExcite](#t1)

[1.2. ConvBnAct](#t2)

[1.3.GhostModule](#t3)

[1.4. GhostBottleneck](#t4)

[1.5. GhostNet](#t5)

### 1. [Backbone](https://so.csdn.net/so/search?q=Backbone&spm=1001.2101.3001.7020)

作为一个着眼于边缘平台部署，尤其是针对 CPU 型设备的网络，NanoDet 之前自然选择的是使用深度可分离卷积的轻量骨干网络。

这里我们主要介绍默认的 Backbone：[GhostNet](https://arxiv.org/abs/1911.11907 "GhostNet"), 这是一个由华为提出的轻量骨干网络, 关于 [GhostNet](https://so.csdn.net/so/search?q=GhostNet&spm=1001.2101.3001.7020) 的详解请戳: 占位符。此模块提供了预训练权重下载，并将结构封装成了一个类。

ghostnet.py 这个文件被放在仓库中的 nanodet/model/backbone 下。

#### 1.0. _make_divisible()

```
# _make_divisible()是一个用于取整的函数,确保ghost module的输入输出都可以被组卷积数整除
# 这是因为nn.Conv2d中要求groups参数必须能被输入输出整除，具体请参考深度可分离卷积相关的资料
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
```

#### 1.1. SqueezeExcite

```
class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        activation="ReLU",
        gate_fn=hard_sigmoid,
        divisor=4,
        **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        # channel-wise的全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1x1卷积,得到一个维度更小的一维向量
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        # 送入激活层
        self.act1 = act_layers(activation)
        # 再加上一个1x1 conv,使得输出长度还原回通道数
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
​
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        # 用刚得到的权重乘以原输入
        x = x * self.gate_fn(x_se)
        return x
```

这个模块来自 [SENet](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf "SENet"), 介绍请戳笔者之前介绍 vision attention 的博客:[CV 中的注意力机制_HNU 跃鹿战队的博客 - CSDN 博客](https://blog.csdn.net/NeoZng/article/details/122663266?spm=1001.2014.3001.5502 "CV中的注意力机制_HNU跃鹿战队的博客-CSDN博客")。利用额外的全局池化 + FC+channel-wise multiply 构建 SE 分支，这能够用来捕捉通道之间的相关性，给予重要的通道更大的权重。

#### 1.2. ConvBnAct

```
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, activation="ReLU"):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layers(activation)
​
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
```

这其实就是卷积、批归一化和[激活函数](https://edu.csdn.net/cloud/ml_summit?utm_source=glcblog&spm=1001.2101.3001.7020)的叠加，这三个结构几乎是现在深度网络的构成单元的标准配置了，写成一个模块方便后面多次调用。

#### 1.3.GhostModule

```
class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, activation="ReLU"
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        # 确定特征层减少的比例,init_channels是标准卷积操作得到
        init_channels = math.ceil(oup / ratio)
        # new_channels是利用廉价操作得到的
        new_channels = init_channels * (ratio - 1)
​
        # 标准的conv BN activation层,注意conv是point-wise conv的1x1卷积
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            act_layers(activation) if activation else nn.Sequential(),
        )
​
        # ghostNet的核心,用廉价的线性操作来生成相似特征图
        # 关键在于groups数为init_channels,则说明每个init_channel都对应一层conv
        # 输出的通道数是输入的ratio-1倍,输入的每一个channel会有ratio-1组参数
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            # BN和AC操作
            nn.BatchNorm2d(new_channels),
            act_layers(activation) if activation else nn.Sequential(),
        )
​
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # new_channel和init_channel是并列的关系,拼接在一起形成新的输出
        out = torch.cat([x1, x2], dim=1)
        return out
```

这个模块就是 GhostNet 的关键了，在了解 GhostNet 所谓的”廉价操作 “即 cheap_operation 之前，你需要知道组卷积(group conv) 和深度可分离卷积 (depth-wise separable conv) 的概念。首先对上一个特征层的输入进行标准卷积, 生成 init_channels 的特征; 随后将此特征进行分组卷积, 并将 groups 数取得和输入的 channel 数相同（每一个 channel 都对应一个单独的卷积核）, 这样就可以尽可能的降低参数量和运算量, 开销非常小.

#### 1.4. GhostBottleneck

GhostBottleneck 就是 GhostNet 的基本架构了, GhostNet 就由数个 GhostBottleneck 堆叠而成，对于 Stride=2 的 bottleneck 在两个 Ghost module 之间增加了一个深度可分离卷积作为连接。

![](https://i-blog.csdnimg.cn/blog_migrate/0770b971c5697cba677639cc609d0fc8.png)

ghost bottle nect 的结构，分为 stage 内的 stride=1 和 stage 间的 stride=2

```
class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE"""
​
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        activation="ReLU",
        se_ratio=0.0,
    ):
        super(GhostBottleneck, self).__init__()
        # 可以选择是否加入SE module
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
​
        # Point-wise expansion
        # 第一个ghost将会有较大的mid_chs即输出通道数
        self.ghost1 = GhostModule(in_chs, mid_chs, activation=activation)
​
        # Depth-wise convolution
        # 对于stride=2的版本(或者你自己选择添加更大的Stride),两个GhostModule中间增加DW卷积
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)
​
        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
​
        # Point-wise linear projection
        # 最后的输出不添加激活函数层,并且会使用一个较小的out_chs以匹配short cut连接的通道数
        self.ghost2 = GhostModule(mid_chs, out_chs, activation=None)
​
        # shortcut
        # 最后的跳连接,如果in_chs等于out_chs则直接执行element-wise add
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        # 如果不相等,则使用深度可分离卷积使得feature map的大小对齐
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
​
    def forward(self, x):
        # 保留identity feature,稍后进行连接
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # 如果stride>1则加入Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x
```

Ghost module 中用于生成复杂特征的卷积是 1x1 的 point-wise conv，对于 Stride=2 的 bottleneck 来说又有一个 stride=2 的 DW，那么就可以将前者就和后者看作是**构成了一组深度可分离卷积**，只不过 Ghost module 生成 ghost feature 的操作大大降低了参数量和运算量。若启用了 has_se 的选项，则会在两个 ghost module 之间加入一个 SE 分支。

#### 1.5. GhostNet

讲解完了基本的模块之后，我们就可以利用上述的 GhostBottleneck 来构建 GhostNet 了：

![](https://i-blog.csdnimg.cn/blog_migrate/69fe40727792d4c15ce524c566ca263c.png)

 GhostNet 原文中整个 backbone 的结构，#exp 是 bottleneck 中通道扩展的倍数，#out 是当前层的输出通道数

​

_#exp_ 代表了在经过 bottleneck 中的第一个 Ghost module 后通道扩展的倍数，通道数随后会在同一个 bottleneck 中的第二个 ghost module 被减少到和该 bottleneck 中最开始的输入相同，以便进行 res 连接。_#out_ 是输出的通道数。可以发现，Stride=2 的 bottleneck 被用在**两个不同的 stage 之间**以改变 feature map 的大小。

为了用作检测网络，删除最后用于分类的 FC，并从 stage4、6、9 分别取出 stage 的输出作为 FPN 的输入。若需要追求速度，可以考虑进一步减少每个 stage 的层数或是直接砍掉几个 stage 也无妨。