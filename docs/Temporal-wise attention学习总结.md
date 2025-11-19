## Abstract  
  
由于事件（events）通常是稀疏和非均匀的，并且具有$μs$的时间分辨率，所以如何有效和高效地处理时空事件流具有很大的价值。SNN具有从时间流中提取有效时空特征的潜力。将耽搁时间聚合为更高时间分辨率的帧时，由于事件的稀疏和非均匀，SNN模型无法注意到串行帧具有不同的信噪比，这干扰了SNN的性能。  
  
本文提出一种**基于时间的注意力SNN（TA-SNN）模型**，用于学习处理事件流的基于帧的表示。即将注意力概念扩展到时间输入，以在训练阶段判断每帧对最终决策的重要性，并在推理阶段丢弃不相关的帧。  
  
## Introduction  
#### DVS摄像机  
  
- 与产生固定低速率同步帧（通常小于每秒60帧）的传统摄像机不同，**DVS摄像机以极高的事件速率（每秒1M到1G事件）对每个像素的亮度变化的时间、位置和极性进行编码**。相比之下，**DVS摄像机需要更少的资源（事件的稀疏性）；$μs$时间分辨率（TR）可以通过产生高速率事件来避免运动模糊；具有高动态范围（140dB与传统摄像机的60dB）**。遇到需要低延迟、低功耗和可变照明稳定性的视觉任务时，DVS摄像机优势突出，已用于告诉目标跟踪、自动驾驶、SLAM、低延迟交互等任务。  
  
- DVS摄像机记录的事件流信息在时间维度上通常是冗余的，这是由高TR和不规则动态场景变化引起的。这使得事件流信息几乎不可能通过基于密集计算的深度神经网络（DNN）直接处理。额外的数据预处理不可避免地削弱了低延迟和事件节能的优势。SNN具有独特的事件触发计算特性，以几乎无延迟和节能的方式响应事件，较为适合处理事件流信息，但仍然有很多优化空间。  
  
#### 文章亮点  
  
- 提出了**TA（temporal-wise attention）SNN**以获得不同时间事件信息的统计特征，生成attention分数，然后对事件信息进行加权，以低延迟、低功耗和高性能进行端到端的训练和推理。  
- 提出了SNN的**IAP（input attention pruning）方法**，即推理阶段在TA模块中使用阈值得到二元（0和1）注意力分数。与使用全输入的方法相比，该方法有相似或更好的性能，为处理事件流信息任务的节能方面带来新思路。  
- 引入了**RCS（random consecutive slice）方法**，以充分利用采样数据。  
- 在LIF模型和LIAF模型上应用本文的方法都取得了比较好的效果。  
  
![Pasted image 20240315152650.png](../static/images/Pasted%20image%2020240315152650.png)  
  
## Methods  
  
LIF模型和LIAF模型可以用下面的式子表示：  
$$\begin{cases}  
U^{t,n}=H^{t-1,n}+g(W^n,X^{t,n-1})\\  
Z^{t,n}=f(U^{t,n}-u_{th})\\  
H^{t,n}=(e^{-\frac{dt}{\tau}}U^{t,n})\circ(1-Z^{t,n})\\  
X^{t,n}=\begin{cases} Z^{t,n} & \text{for LIF}\\  
ReLU(U^{t,n}) & for LIAF\end{cases}  
\end{cases}$$  
其中$g(*)$表示卷积或全连接操作，$f(*)$是Heaviside step function，$e^{-\frac{dt}{\tau}}$反映了膜电位的泄露情况，$\circ$是Hadamard乘积（按元素相乘）。  
### Temporal-wise Attention for SNNs  
  
**TA模块的目标是估计每个帧的显著性。该显著性分数不仅基于当前时间步的输入统计特征，还要考虑相邻帧的信息**。在时间方向上应用 squeeze step 和 excitation step 来实现上述两点。$t$时刻第$n$层的空间输入张量为$X^{t,n-1} \in R^{L×B×C}$，其中$C$为通道大小。  
  
**squeeze 步骤计算事件数的统计向量**，统计向量$s^{n-1} \in R^T$在时间步$t$的值为：  
$$s^{n-1}_t=\frac{1}{L×B×C} \sum^{C}_{k=1}\sum^{L}_{i=1}\sum^{B}_{j=1}X^{t,n-1}(k,i,j) $$  
**excitation 步骤中，$s^{n-1}$通过两层全连接网络进行非线性映射，以获得不同帧之间的相关性**，即**得分向量（score vector）**：  
$$d^{n-1}=  
\begin{cases}  
\sigma (W^n_2 \delta(W^n_1(s^{n-1}))) &\text{training,}\\  
f(\sigma(W^n_2\delta(W^n_1s^{n-1}))-d_{th}) &\text{inference,}\\  
\end{cases}$$  
$\delta$和$\sigma$分别是ReLU和sigmoid激活函数，$W^n_1 \in R^{\frac{T}{r} ×T}$和$W^n_2 \in R^{T × \frac{T}{r}}$是可训练的参数矩阵，可选参数$r$被用于控制模型的复杂度，$f(·)$是Heaviside step function，$d_{th}$是attention阈值。  
  
在训练阶段使用得分向量来训练一个完整的网络。作为可选操作，在推理阶段，我们舍弃低于$d_{th}$的无关帧，将其他帧的注意力评分设置为1。最终将$d^{n-1}$作为输入的score vector，所以在$t$时间步的最终输入为：  
$$\widetilde X^{t,n-1}=d^{n-1}_{t}X^{t,n-1} $$  
然后，TA-LIF和TA-LIAF的膜电位行为遵循：  
$$U^{t,n}=H^{t-1,n}+g(W^n,\widetilde X^{t,n-1}) $$  
**TA模块可以看作是一个自注意力函数，区别在于得到的统计向量与事件数直接相关**。  
### RCS (random consecutive slice) Method  
  
忽略硬件中的时间消耗，基于事件的系统延迟$t_{lat}$仅取决于$dt$和$T$，即$dt × T$。本文**在训练阶段**采用了**RCS的数据增强方法**，即**随机选择一个时刻$t_0$作为起点，然后聚合后面的连续帧**。  
  
**在测试阶段，采用投票机制**，即对于给定的$dt$和$T$，将一个完整的事件流划分为连续的10个crops，每个crop的长度为$t_{lat}$，将所有crop的结果累加即可得到最终的测试结果。如果帧数小于$10×t_{lat}$，则采用overlap的方法。例如，使用两个crop，事件流总时间为$30ms$，$t_{lat}$等于$20ms$，crop将覆盖重复的范围，如$[0ms: 20ms]$和$[10ms:30ms]$。  
  
![Pasted image 20240316132241.png](../static/images/Pasted%20image%2020240316132241.png)  
  
## Experiment  
  
### 数据集  
  
DVS128 Gesture、CIFAR10-DVS、Spoken Heidelberg Digits（SHD, 一个音频分类数据集）  
  
### 对于不同数据集的网络结构  
  
![Pasted image 20240316132446.png](../static/images/Pasted%20image%2020240316132446.png)  
  
### TA不同位置的实验  
  
- **无TA、仅在输入层插入TA、在输入层之外的整个网络插入TA、将TA插入整个网络**  
- 不同的时间步$T=\{30,60,90,120,150,180\}$，$dt=1ms$进行试验  
- 数据集为DVS128 Gesture  
![Pasted image 20240316132901.png](../static/images/Pasted%20image%2020240316132901.png)  
结论：在大多数情况下，仅在输入层插入TA和在输入层之外的整个网络插入TA可以提高性能，且后者能达到最佳精度，但将TA插入整个网络会导致不稳定的结果。当$T$处于前半范围内时，提高$T$可以略微提高精度，但进一步扩大$T(T>120)$作用不大。  
  
### IAP（input attention pruning）实验  
  
- IAP：在推理阶段舍弃注意力分数较低的帧以进行节能  
- 本文进行了IRP（输入随机剪枝）和IAP的对比试验  
- 数据集为DVS128 Gesture  
![Pasted image 20240316133611.png](../static/images/Pasted%20image%2020240316133611.png)  
结论：IRP的精度随剪枝比例的增加而近似单调下降，而IAP在剪枝比例小于0.5时，随比例增加，精度不会降低。剪枝比例在0.5时，IAP的精度约为89%，而IRP为785。IAP的最佳剪枝比例与时间步长$T$有关。使用TA模块能在降低功耗的同时得到与全输入近似甚至更高的精度。  
  
### RCS和TA的消融实验  
  
由前面的实验结果可知，在输入层之外的整个网络插入TA（S3）稳定性较好。  
  
消融实验设置$dt=\{1,5,10,15,20,25\}$，固定$T=60$，实验结果如下：  
  
![Pasted image 20240316134049.png](../static/images/Pasted%20image%2020240316134049.png)  
  
结论：TA在任何条件下都能工作。当$dt$较小时，RCS对精度有很大提高，当$dt$较大时，效果减弱。对于LIF，RCS和TA可以很好地协同工作，准确率最高为95.49%；对于LIAF，当$dt$较大时（$dt \in \{15,20,25\}$），RCS有负面影响。  
  
  
  
  
  
  
