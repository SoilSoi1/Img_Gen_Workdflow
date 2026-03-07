# 毕业内容一的记录
## 前言
这一章的标题为***基于扩散模型的双极板表面数据集的扩充与优化***，本质上我是想扩充数据集的数量，因为拍板子效率很低。  

一开始选择的是基于GAN的图像生成方法，但是文献中效果并不理想，后来转向了扩散模型。但是计划的时候没有想到扩散模型的调试难度这么大，本身是为了后面预测模型的训练做准备的，结果花了太多时间在图像生成上面。  

但是，既然已经选择了这个方法，哪怕最后做不下去，好歹也要有个理由吧，所以这也是我选择开始记录文档的原因。毕竟每一次调试都有不可预测性，无论计划得多完整，实际操作中总会遇到各种各样的问题。
所以这里我会记录下每一次调试的内容，方便后续回顾和总结。   

---
立个flag，就当作为我的小论文或者大论文做记录吧。

---
### 爱因斯坦求和约定
***Einstein summation convention*** 

插播一段：爱因斯坦求和约定与pytorch相关的内容

首先，非常简单地介绍一下爱因斯坦求和约定的内容。
既然是“约定”，那么它肯定是适用于很多种特殊情况下的一种简化方式，那么对于深度学习中的计算来说，我们最需要的其实只是所谓张量计算（按照我的理解就是矩阵运算，虽然本质上还是有区别的，就是这么理解）。

最常见的：

设：$A \in \mathbb{R}^{m\times n} , B\in \mathbb{R}^{n\times p}$

在普通的矩阵乘法中，`A`和`B`两个矩阵的积表示为

$$
C_{ij}=\sum_{k=1}^n A_{ik}B_{kj}
$$
如果当某个指标在一个项中 恰好出现两次（一次上标、一次下标，或同一位置的两次），则默认对该指标求和：
$$
C_{ij}=A_{ik}B_{kj}
$$

如上述公式，原本对 $k$ 从1到n进行求和的操作被简化掉了求和符号，同时这也是在矩阵乘法中最常见的一种形式，即前者列数等于后者行数。
至于其他情况下的约定，这里暂不赘述。

下面将介绍在Pytorch中，如何利用爱因斯坦求和约定简化张量运算。

```python
# Example
import torch

a = torch.rand(2,3)
b = torch.rand(3,4)
c = torch.einsum("ik,kj->ij", [a, b])
# 等价操作 torch.mm(a, b)
```
爱因斯坦求和约定在torch中的计算语法为`einsum(equation, *operands) -> Tensor`。
其中传入参数`equation`代表着例子中的`ik,kj->ij`，它表示了输入输出张量的维度。`equation` 中的箭头左边表示**输入张量**，以**逗号**分割每个输入张量，箭头右边则表示**输出张量**。表示维度的字符只能是26个英文字母 'a' - 'z'。

而`*operands`是一个可变参数，表示实际的输入张量列表，其数量要与`equation`中的输入数量对应。同时对应每个张量的子`equation`的字符个数要与张量的真实维度对应，比如 "ik,kj->ij" 表示输入和输出张量都是两维的。

再介绍两个概念：**自由索引（Free indices）** 和 **求和索引（Summation indices）**：

- 自由索引，出现在箭头右边的索引，比如上面的例子就是 i 和 j；
- 求和索引，只出现在箭头左边的索引，表示中间计算结果需要这个维度上求和之后才能得到输出，比如上面的例子就是 k；

特殊规则：
- equation 可以不写包括箭头在内的右边部分，那么在这种情况下，输出张量的维度会根据默认规则推导。就是把输入中只出现一次的索引取出来，然后按字母表顺序排列，比如上面的矩阵乘法`ik,kj->ij`也可以简化为`ik,kj`，根据默认规则，输出就是`ij`与原来一样；
- equation 中支持`...`省略号，用于表示用户并不关心的索引，比如只对一个高维张量的最后两维做**转置**可以这么写:
    ```python
    a = torch.randn(2,3,5,7,9)
    # i = 7, j = 9
    b = torch.einsum('...ij->...ji', [a])
    # b = torch.einsum('...ij->...ji', a)同理
    ```
    


---

*\*这条更新于12月16日，具体内容为研究内容一的整体规划的整理，整理的目的是需要在1月30日之前完成所有关于图像扩充部分的内容记录*
## 研究内容一的大纲整理
它们之间间隔一天或者两天，用来思考和规划具体完成任务的方式，思路要明确。
### 12月17日-12月26日
完成两个任务：**评估质量**和**拍板子**

评估质量指的是完成所有指标，包括FID | KID | T-sne Distribution（暂时这么多）的**代码编写和测试矫正**；

拍板子指的是把科创大部分板子给拍完，我觉得可以提升一些效率，比如一张板子只拍40张，甚至25张。

这两步尽量同时完成，因为可能是唯一一对不需要真实板子全部收集齐的组合了，争取一个星期结束战斗。

难点可能在于需要花点时间去找*什么是“数据分布”*，并实现某种可视化形式。

#### 详细规划
- 对于FID和KID来说，需要明确的是*input*与*output*接口。
最基本的是一定会有**输入**的两组分别包含有truth和gen图片的文件夹，**输出**有FID\KID的值，总的来说，参数规划如下：

|input|output|
|:------:|:---:|
|Data_1||
|Data_2||
|他们的默认参数||


### 12月29日-1月20日
这段时间完成传统图像增强、VAE、LDM和**改进LDM**的实现（先完成这个，即可确定模型，当然，内定为改进LDM），以及基于它们的图像生成（与真实样本数量的不同比例对比），最后基于上一部分的任务对它们进行质量评估，所有记录数据都在这里完成记录。

***（待补充细节）***

### 1月22日-1月30日
最后一个阶段，把所有生成的图像代入resnet-18或者50，验证图像生成的有效性，比如用假数据训练预测真实样本，或者反过来预测等

***（待补充细节）***


---
## 开始尝试DDPM
*Denoising Diffusion Probabilistic Models*，简称DDPM，中文名为去噪扩散概率模型。*.      

其实一开始对扩散模型的原理不是很了解，DDPM应该是整个扩散模型中最基础的一个版本了吧。原论文连接: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). 

我参考的实现是这个：[DDPM-PyTorch](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/)，这个版本代码非常清晰，没有任何多余的实现。
分为无条件的DDPM和条件DDPM两部分，对于我的任务来说，我不需要有条件任务，因为我的任务只是二分类，我可以直接在文件层面隔开，不然用有条件任务还需要额外的标签输入。   

先介绍一下DDPM的原理吧。DDPM的核心思想是通过逐步添加噪声将数据分布转换为简单的高斯分布，然后通过学习一个反向过程来逐步去除噪声，从而生成新的数据样本。这个过程可以分为两个阶段：前向扩散过程和反向去噪过程。

<figure>
    <img src="src/ddpm_schematic.png">
    <figcaption style="text-align:center;font-family:Times"> Fig 1. The schematic diagram of DDPM</figcaption>
</figure>

### 前向扩散过程
首先，定义一个表示添加噪声程度的参数 $\beta_t$ ，该参数的值越大，表示添加到数据中的噪声就越大，在原论文中通常呈递增，范围为$1\times10^{-4}$ ~ $2\times10^{-3}$。  

此外，再定义一个参数 $\alpha_t$，$\alpha_t=1-\beta_t$，主要是为了方便后续计算。  

这里补充一个概念就是马尔可夫链[^1]。  

那么，基于中间任意时刻的一张图片$x_{t-1}$，如何得到 $x_t$ 呢？  

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}z_1 \tag {1-a}
$$
如以上公式所示，其中$z_1$是一个属于标准高斯分布$N(0,1)$的随机噪声。
其中可能令人困惑的是，为何该公式中有两种噪声？
其实$\alpha$（尽管来源于$\beta$）很明显是一个人为设定的值，而我们的目的是从纯噪声去预测图像，如果只有人为设定好的值，那还谈什么预测呢？所以，需要引入随机噪声$z$去制造一个未知的量（深度学习本质的目的），以便神经网络去拟合噪声的特征，从而还原出图像。  

也许还有疑问：为什么要预测噪声？其实也很简单，就是利用这一点点随机噪声，在大部分还符合原分布的情况下，去创造新的图像。

下面进入公式推导阶段。  

如Fig 1.所示，从原始输入图片$x_0$到$x_T$的过程中，总共要经历$T$次加噪过程。在一般的实践中，$T$的大小通常在200-1000次，算力足够的情况下甚至更多。训练模型的过程中，如果发生这么多次加噪过程，每次都要重新从$0$一直算到目标的$T$，非常消耗资源。幸运的是，有一种方法可以节省重复计算带来的资源浪费问题，即不需要每次频繁迭代，即可快速得到目标步长所需要的加噪图像：  

首先，考虑到公式1-a是一种迭代方式，很自然地就可以想到迭代$x$换元的方法，即  
$$
x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}z_2 \tag {1-b}
$$
其中$z_2$也服从标准正态分布，带入公式1-a中的$x_{t-1}$，即     
$$
x_t=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\left(\sqrt{1-\alpha_t}z_1+\sqrt{\alpha_t(1-\alpha_{t-1})}z_2\right) \tag {1-c}
$$
n个标准正态分布相加是可以合并的，公式3中的两个标准正态分布的期望和方差可以表示为:
$$
\begin{cases}
\mu  =0 \\
\sigma^2=\sigma_1^2+\sigma_2^2=\alpha_t(1-\alpha_{t-1})+(1-\alpha_t)=1-\alpha_t\alpha_{t-1}
\end{cases}
$$
$$
\Rightarrow x_t=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}z';
z'\sim N(0,1)
\tag {1-d}
$$
$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}z_1 \tag {1-a}
$$
将公式1-d与公式1-a对比，可以看到，经过一次推导，就会发现该式是有迭代规律的，原来$\alpha_t$的位置都变为了$\alpha_t\alpha_{t-1}$，所以可以大胆猜测(其实是必然)，我们完全可以通过$x_0$一次性推出任何时步的$x_t$，当然也包括最终的$x_T$:
$$
x_T=\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}z 
\tag {1-e}
$$
其中$\bar{\alpha_t}=\prod_{t=0}^t \alpha_i$。

这样，实际代码的实现过程中，就可以先算出包含所有$\alpha_t$的列表，需要时直接根据对应的索引获得即可，大幅节省了训练效率。
### 反向采样过程
这里可以用概率论的知识来解释。定义：  
$$
Forward:q(x_t|x_{t-1})
$$
$$
Sampling: p(x_{t-1}|x_t)
$$
由贝叶斯定理[^2]得：
$$
p(x_{t-1}|x_t)=q(x_t|x_{t-1})\frac{q(x_{t-1})}{q({x_t})}
$$
由马尔可夫链的性质（$x_0$与$x_t$相互独立）得：
$$
p(x_{t-1}|x_t)=q(x_t|x_{t-1})\frac{q(x_{t-1}x_{0})/q(x_0)}{q({x_tx_0})/q(x_0)}
$$
$$
\implies =q(x_t|x_{t-1})\frac{q(x_{t-1}|x_0)}{q({x_t}|x_0)}
\tag {2-a}
$$
由正向传播中式(1-a)和(1-e)可知：
$$
\begin{cases}
q(x_t|x-0)=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}z\sim(\sqrt{\bar{\alpha}_t}x_0,1-\bar{\alpha}_t)\\
q(x_{t-1}|x-0)=\sqrt{\bar{\alpha}_{t-1}}x_0+\sqrt{1-\bar{\alpha}_{t-1}}z\sim(\sqrt{\bar{\alpha}_{t-1}}x_0,1-\bar{\alpha}_{t-1})\\
q(x_t|x_{t-1})=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}z\sim N(\sqrt{\alpha_t}x_{t-1},\sqrt{1-\alpha_t})
\end{cases}
$$
将三式代入式（2-a），并结合正态分布概率密度函数得：
$$
p(x_{t-1}|x_t)\propto \exp{(-\frac{1}{2}\left(
    \frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{\beta_t}+
    \frac{(x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1-\bar{\alpha}_{t-1}}-
    \frac{(x_{t-1}-\sqrt{\bar{\alpha}_t})^2}{1-\bar{\alpha}_t}
\right))}\\
\propto\exp{(-\frac{1}{2}\left(
    (\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar{\alpha}_{t-1}})x_{t-1}^2-
    (\frac{2\sqrt{\alpha_t}}{\beta_t}x_t+\frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}})x_{t-1}+C
\right))}\\
\implies \mu=\frac{1}{\sqrt{\alpha_t}}(
    x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{z_\theta}
)
$$
原文如此描写：
$$
\nabla_\theta \lVert z-z_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}, t) \rVert^2
$$
至此，大概的DDPM推导就结束了。  
下面我将会尽可能详细记录一下所有实验的过程。

### 试跑通阶段
同样强调一下，代码是基于[DDPM-PyTorch](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/)这个仓库实现的。

该仓库的实现十分明确，分为三个核心代码`Diffusion.py` `Model.py` `Train.py`以及一个主函数`Main.py`

基于我的任务，以该仓库为基础，做了一些改动：

#### 1.
由于所需训练的数据为伪彩图像对于在RGB空间中占据了同等信息量下灰度空间的三倍计算，所以在`Main.py`中的`modelConfig`字典中，增加了一个传入数据的通道数`input_channel`，用来区分传入的数据通道数，以避免在U-Net中存在不匹配的问题；  

该改动还包括在`Model.py`中的`UNet`类中传入属性`self.input_channel`，替换原来默认为3的传入通道数。
该改动还包括在`Train.py`中的`eval()`中更改了传入通道数。
#### 2.
由于经常性地需要断点重续，所以在`Train.py`中增加了记录epoch的功能，函数名为`epoch_file`。
#### 3.
在`Train.py`中，使用了之前在预测模型的留下的`dataset.py`作为数据预处理的方式。但是这样一个省事的方式，结果导致了后来一个严重的有关于**归一化**的问题。

我一开始不知道在扩散模型中，甚至说在U-Net中，几乎有个默认的，却又很适合一般工程工作的潜在约定，即输入图像需要归一化至$[0,1]$或者$[-1,1]$，因为标准高斯噪声的对称分布性，后者更加常用。

而我犯的一大错误在于，我用了在分类任务重最常用的基于ImageNet的归一化策略`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`，这导致了在去噪扩散任务中，输入图像的像素值范围被归一化至了$[-1.996,2.449]$附近，既不符合对称，也不符合与标准高斯分布相匹配的方差数量级。**因此**，我改动了这个归一化策略为`    transforms.ToTensor(),
transforms.Normalize(mean=[0.5], std=[0.5])`
，原本分布在$[0,255]$的像素值在`transforms.ToTensor()`之后，被放缩成$[0,1]$，再经过`transforms.Normalize(mean=[0.5], std=[0.5])`之后，被放缩为$[-1,1]$，从而达到去噪扩散模型所要求的、合理的输入数据范围。

在介绍完改动区域后，我会尽可能详细介绍一下为什么改动以及如何改动

#### 4.
由于需要对生成图片进行自定义保存和修改，将`Train.py`中的`eavl()`最后的生成图像保存修改了一下逻辑，由直接保存改为返回未保存图像`return sampledImgs`。

#### 5.
我自己写了一个基于DDPM快速生成目标数量图像并保存的脚本，脚本命名为`gen_quick.py`，内容如下：
```python
def sampling(num_pic, saved_dir):
    for i in range(num_pic):
        sampled_img = eval(modelConfig)
        save_image(sampled_img, f"{saved_dir}/sampled_img_{i}.png", nrow=modelConfig["nrow"])

if __name__ == '__main__':
    modelConfig["state"] = "eval"
    num_pic = 100
    weight = modelConfig["test_load_weight"].split("/")[-1][:-3]
    os.mkdir(f"./outfig/ddpm/{weight}", exist_ok=True)
    saved_dir = f"./outfig/ddpm/{weight}"

    s_t = time.time()
    sampling(num_pic, saved_dir)
    e_t = time.time()
    print(f"采样 {num_pic} 张图像总共用时: {e_t - s_t} 秒")
```
主要功能还是比较清晰的，封装单词生成一张图像的函数，并在主函数中确定生成的数量和保存路径，最后记录总采样时间。

#### 6.
更改了checkpoint的保存逻辑，每500个epoch才保存一次，但是次数的设计取决于迭代次数，目前1轮epoch代表了100次迭代，所以500个epochs有50000次迭代，记为ckpt_50k，以此类推。
```python
torch.save(net_model.state_dict(), os.path.join(
    modelConfig["save_weight_dir"], 'last_ckpt.pt'))
if e % 500 == 0 and e != 0:
    torch.save(net_model.state_dict(), os.path.join(
        modelConfig["save_weight_dir"], f'ckpt_{e}epoch.pt'))
epoch_file(e, f'{modelConfig["save_weight_dir"]}/output.txt')
```

### 关于去噪模型中归一化的问题
下面我用信噪比(SNR)用并不严谨的方式来简要解释一下。  

首先借用一个比喻：
> 你和一个朋友在咖啡店聊天，但是周围环境很吵。
此时：
> - 你们谈话的声音 $\to$ 信号
> - 周围的音乐、人声 $\to$ 噪声
>
> 声音 $\gg$ 噪声 $\to$ 听得很清晰（高SNR）   
> 声音 $\sim$ 噪声 $\to$ 有点吃力  
> 声音 $\ll$ 噪声 $\to$ 完全听不清（低SNR）

应该是比较清晰的比喻，现在再搬回扩散模型。  

在扩散模型中，再次重复*式1-e*：
$$
x_T=\sqrt{\bar{\alpha_t}}x_0+\sqrt{1-\bar{\alpha_t}}z\\
\sqrt{\bar{\alpha_t}}x_0\implies 信号\\
\sqrt{1-\bar{\alpha_t}}z\implies 噪声
$$
可以看到，左边部分为残留的信号信息，右边为逐渐添加的噪声部分。  

这个例子特别好，我也是一下就理解了为什么要定义$\alpha$这样一个量，可以直观地感受到在逐渐加噪的过程中，原有信号的剩余量。

准确来说，在扩散模型中，信噪比SNR可以被定义为：
$$
{SNR}=\frac{\bar{\alpha_t}}{1-\bar{\alpha_t}}
$$
容易看出SNR和时步t的关系：当t为0时，即图像还未被加噪时，SNR$\gg$1；当t为T时，即图像完成最后一步加噪时，SNR$\ll$1。

有一句话总结地特别好：
**<center> t本身其实没物理意义，SNR才是“真正的进度条” </center>**

现在，可以回过头来看一下，为什么归一化至除了$[-1,1]$都是不支持的，毕竟可能的疑问是，高斯噪声明明是属于$\mathbb{R}$，按道理来说没有什么不行的。

现在假设将图像的域扩大至$[-2,2]$，对于二维图像来说，相比较于$[-1,1]$的图像来说，信息含量增大了很多，但是与此同时，噪声却还维持在原有的水平，结果就是，在同样的时步t下，图像比原来还清晰许多，那么整个扩散过程的时间轴就被拉长了。

举个极端的例子就是
- 如果信号无限大，那么无论加多少噪声都无法淹没原图
- 如果信号很小，那么噪声很轻易就会覆盖掉原图

现在好像解决了选择合适的归一化，是否在扩散模型的具有重要意义的问题。

但，**为什么是[-1,1]呢？**

首先给出结论：
**<center>[-1,1]并不是数学上最精确的范围，而是它刚好落在扩散模型需要的合适的尺度</center>**

扩散模型的需要：
- 数据均值为0
- 方差为$O(1)$

将一个数据映射至$[-1,1]$之后，且假设图像是均匀的：
- $\mu=0$
- $\sigma\approx\frac{1}{3}$

如果将数据映射至$[-2,2]$:
- $\mu=0$
- $\sigma\approx\frac{4}{3}$

结合刚才的结论，可以看出，并不是其他的归一化范围不能使用，而是考虑了众多的通识，是图像处理中约定俗成的数字。
理论上来说，在数据均匀时，$[-\sqrt{3},\sqrt{3}]$理应是最合适的，它让方差正好为1。
但实际情况并不是这样，一是处理的图像并不符合均匀图像，二是历史原因，包括GAN、U-Net在内的大多数情况下，都选择将数据预处理至$[-1,1]$，如果重新选用归一化范围，那么就相当于为了一个无关紧要、无法确定是否有效的数字连带更改了其余所有参数，这是不值得的。

总结来看，使用$[-1,1]$是深度学习中最适合的尺度。

---

### 试跑通的结果与参数
在成功可以生成像样的图片后，模型参数为：
```python
    "state": "eval", # or eval
    "epoch": 2000,
    "batch_size": 2,
    "T": 400,
    "channel": 64,
    "channel_mult": [1, 2, 3, 4],
    "attn": [2],
    "num_res_blocks": 1,
    "dropout": 0.15,
    "lr": 2e-5,
    "multiplier": 2.,
    "beta_1": 1e-4,
    "beta_T": 0.04,
    "img_size": 512,
    "grad_clip": 1.,
    "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
    "training_load_weight": "last_ckpt.pt",
    "save_weight_dir": "./Checkpoints/",
    "test_load_weight": "grey/ckpt_50k.pt",
    "sampled_dir": "./SampledImgs/",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "SampledNoGuidenceImgs.png",
    "nrow": 8,
    "input_channel":1
```
综合考量下总时步T设定为400，$\beta$设定范围$1\times10^{-4}\sim4\times10^{-2}$，与原文不太一样。

为了衡量生成图像的质量，基于clean-fid库，简单完成了一个计算fid的脚本。
但是fid、kid等评估值的计算仍然需要其他方面的考量，例如搞清原理，fid似乎是基于某个数据集的权重计算的，在我的数据集上表现得绝对值很高。虽然看绝对值是无意义的，但是这是一个（对于写论文来说）风险，我需要尽可能避免它。

下面是基于该脚本简单计算的FID值：

|Weights|FID|
|:----:|:----:|
|ddpm_50k|243.7245|
|ddpm_200k|161.4483|
|ddpm_400k|163.1784|
|ddpm_600k|144.4969|

<figure>
    <img src="src/sampled_pic1.png">
    <figcaption style="text-align:center;font-family:Times"> Fig 2. (a)Truth;(b)Pictures generated by DDPM</figcaption>
</figure>

权重与生成的样图（每个权重100张）都保存在`experiments/ddpm/first/`下。

---

## 完整的图像质量评估
我首先搜集了一些适合于基于私有数据集训练的图像生成模型生成图像质量的评估指标：

|指标|维度|输入|
|:---:|:---:|:---|
|FID|综合|（2）生成数据集与真实数据集|
|KID|综合（无偏估计）|（2）生成数据集与真实数据集|
|Precision & Recall|质量、多样性|（2）生成数据集与真实数据集|
|Memorization Check|过拟合检测|（2）生成数据集与真实数据集|
|Intra-LPIPS|图像相似度|（1）生成数据集|
|BRISQUE|图像去噪完整度|（1）生成数据集|
|CAS|在分类器上的精度|（1）生成数据集，但需要提前训练分类器|


我将会从相关文献、数学原理、代码实现三个方面分别描述这些评估图像质量的指标，并在最后以试跑通的生成数据与真实数据为例，做一个pipeline演示。其中预实现的pipeline应该包含有基本的生成图片数量与保存功能。

---

### FID
FID全称Fréchet Inception Distance，来源文献[GANs trained by a two time-scale update rule converge to a local nash equilibrium](http://arxiv.org/abs/1706.08500).
#### 数学原理
FID不是逐像素比较，而是在高维空间中比较。
要做到这一点，FID采用了预训练的Inception-V3，取pool3层的输出，即$f(x)\in\mathbb{R}$.  
取真实图像集：$\{x_i^{(r)}\}\to \{f_i^{(r)}\}$.   
取生成图像集：$\{x_i^{(g)}\}\to \{f_i^{(g)}\}$.  

FID的核心假设：**Inception特征在高维空间中服从多元高斯分布**

所以假设：  
$$
P_r\sim N(\mu_r,\Sigma_r)
$$
$$
P_g\sim N(\mu_g,\Sigma_g)
$$
其中参数再通过样本估计：   
**均值：**
$$
\mu_r = \frac{1}{N_r} \sum_{i=1}^{N_r} f_i^{(r)}, \quad
\mu_g = \frac{1}{N_g} \sum_{j=1}^{N_g} f_j^{(g)}
$$
**协方差：**
$$
\Sigma_r =
\frac{1}{N_r - 1}
\sum_{i=1}^{N_r}
\left(f_i^{(r)} - \mu_r\right)
\left(f_i^{(r)} - \mu_r\right)^T
$$
$$
\Sigma_g =
\frac{1}{N_g - 1}
\sum_{j=1}^{N_g}
\left(f_j^{(g)} - \mu_g\right)
\left(f_j^{(g)} - \mu_g\right)^T
$$

最后，引入**Wasserstein-2 距离**，定义为：

$$
W_2^2(P, Q)=
\inf_{\gamma \in \Pi(P, Q)}
\mathbb{E}_{(x, y) \sim \gamma}
\left[ \|x - y\|^2 \right]
$$

$$
FID=
W_2^2(P_r, P_g)=
\lVert\mu_r - \mu_g\rVert^2+
\mathrm{Tr}
\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^\frac{1}{2}\right))
$$

#### 代码实现
保存在`_fid.py`脚本下，其中函数名为`cal_fid`。  
默认的测试数据路径是`./evaluators/test_data`下，其他评估函数同理。

实现目标：能够读取两个文件夹（真图与假图），并输出得分。

选型考虑：放弃了复杂的原始实现，选择 `clean-fid` 库。因为它解决了不同 Resize 算法带来的结果偏差，对 Diffusion 这种高质量图像生成任务更具参考价值。

在`fid`的`compute_fid`方法中，配置为：
```python
fid_score = fid.compute_fid(
    path_real_images, 
    path_fake_images,  
    mode="clean",               
    device=device,
    num_workers=8 if device == 'cuda' else 0 # For MAC
)
```

---

### KID
KID全称为Kernel Inception Distance，原文为[Demystifying MMD GANs](https://arxiv.org/abs/1801.01401)。该文献提出 KID 的核心贡献为该指标为一种**无偏估计 Unbiased Estimator**，是 FID 的一种替代指标
#### 数学原理
KID是一种用**核方法**来衡量真实图像分布和生成图像分布在 Inception 特征空间中的差异。

和 FID 一样，第一步同样是用 Inception 去提取图像集在高维空间中的特征 $\mathbb{R}^d$。

然后，与 FID 不同的是，KID 不是用“假设两组图像服从高维高斯分布”来衡量分布差异，而是用 **MMD(Maximum Mean Discrepancy)** 最大均值差异。

花了一点时间理解其中的数学原理，这里先介绍 **Mercer定理**。

设存在一种核函数
$$
f(x_i, x_j)
$$
其中$x_i,x_j \in \{x_1, x_2, ... , x_n\}$

定义Gram矩阵（这是一个专有名词）：
$$
Element:a_{ij}=f(x_i,x_j)
$$
$$
Gram=
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{pmatrix}\
=\
\begin{pmatrix}
f(x_1,x_1) & f(x_1,x_2) & \cdots & f(x_1,x_n) \\
f(x_2,x_1) & f(x_2,x_2) & \cdots & f(x_2,x_n) \\
\vdots & \vdots & \ddots & \vdots \\
f(x_n,x_1) & f(x_n,x_2) & \cdots & f(x_n,x_n)
\end{pmatrix}
$$
该矩阵维度为$n \times n$。

如果该Gram矩阵是半正定矩阵，则称f(x)为**半正定函数**。
**<center>Mercer定理：所有半正定函数都可以作为核函数</center>**

接着和FID类似，取Inception特征：
- 取真实图像集：$\{x_i^{(r)}\}\to \{f_i^{(r)}\}$.   
- 取生成图像集：$\{x_i^{(g)}\}\to \{f_i^{(g)}\}$.  

KID中使用的核函数是**多项式核函数**，定义为：
$$k(f_i^{(r)},f_j^{(g)})=\left(\frac{f_i^{(r)} \cdot f_j^{(g)}}{d}+1\right)^3$$

然后，KID的计算公式为：
$$KID = \frac{1}{N_r N_g} \sum_{i=1}^{N_r} \sum_{j=1}^{N_g} k(f_i^{(r)}, f_j^{(g)}) - \frac{1}{N_r^2} \sum_{i=1}^{N_r} \sum_{j=1}^{N_r} k(f_i^{(r)}, f_j^{(r)}) - \frac{1}{N_g^2} \sum_{i=1}^{N_g} \sum_{j=1}^{N_g} k(f_i^{(g)}, f_j^{(g)})$$

含义和FID类似，但是KID是无偏估计，使用了MMD。

#### 代码实现

同样是用clean-fid实现kid的计算，接口和fid的实现基本相同。

```python
kid_score = fid.compute_kid(
    path_real_images,
    path_fake_images,
    mode="clean",
    device=device,
    num_workers=8 if device == 'cuda' else 0
)
```

---



---



[^1]:又称离散时间马尔可夫链，该过程要求具备“无记忆”性质：下一状态的概率分布只能由当前状态决定，在时间序列中它前面的事件均与之无关
[^2]:$$