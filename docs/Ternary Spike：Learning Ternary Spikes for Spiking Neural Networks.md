## Introduction  
  
SNNs adopt spikes to transmit information and thus can convert multiplications of weights and activations to additions, enjoying multiplication-free inference. Furthermore, the event-driven-based computation manner shows higher energy efficiency on neuromorphic hardwares.  
  
However, **SNN's binary spike activation maps suffer the limited information capacity compared to full precious activation maps of the ANN and are unabe to carry enough useful information from membrane potentials in the quantization process of the SNN, thus causing information loss and resulting in accuracy decreasing**. Meanwhile, we also find that **the membrane potential distributions of different layers of an SNN are much different, thus quantizaing these membrane potentials to the same spike values is unnatural**, which is ignored by the prior work.  
  
To solve these problems, we first propose the ternary spike neuron, called **Ternary Spike**. Different from the current way using $\{0,1\}$ spikes, it utilizes the $\{-1,0,1\}$ spikes to transmit information. Furthermore, we also extend the ternary spike to a learnable ternary spike form, which is not limited to $\{-1,0,1\}$, but $\{-\alpha,0,\alpha\}$, where $\alpha$ is layer-wise learnable value. In this way, different layers' neurons will fire different magnitudes of spikes in a learning manner, corresponding to different membrane potential distributions. In the inference phase, the $\alpha$ factor can be folded into the weights by a re-parameterization technique, hence will retain the multiplication-free inference again.  
  
## Related Work  
### Learning Methods of Spiking Neural Networks  
  
There are mainly two routers to obtain high-performance deep SNNs. The first is converting a well-trained ANN to an SNN, called **ANN-SNN conversion**. The principle of the ANN-SNN conversion method is to map the parameters from a pre-trained ANN model to its SNN counterpart, based on the matching of the ANN activation values and the SNN average firing rate. **There are still several inherent deficiencies in ANN-SNN conversion that are difficult to solve**. First, it is limited in the rate-coding scheme and ignores the rich temporal dynamic behaviors from SNNs, thus it cannot handle there neuromorphic datasets well. Second, it usually requires many timesteps to approach the accuracy of pre-trained ANNs. This will increase energy consumption correspondingly, which is contrary to the original intention of SNN's low-power consumption design. Third, the SNN accuracy cannot exceed the ANN accuracy in this paradigm. This will limit the imagination and possibility of the SNN, thus reducing its research values.  
  
Training SNNs directly from scratch is suitable for neuromorphic datasets and can greatly reduce timesteps, even less than 5 in recent work. Hence this kind of model, as another router of SNN learning has received more research attention recently. Besode, the **"hybrid" learning** method, whi h combines the advantages of ANN-SNN conversion method and direct training method, has also drawn much attention recently.  
  
## Methodology  
### Information Loss in Spiking Neural Networks  
  
Given a set, $S$, its representation capability, $\mathcal{R}(S)$ can be measured by the information entropy of $S$, as follows  
$$  
\mathcal{R}(S)=max \mathcal{H}(S)=max(-\sum_{s \in S}p_S(s)logp_S(s)) \tag{6}  
$$  
where $p_S(s)$ is the probability of a sample, $s$ from $S$.   
  
**Proposition 1** *When $p_S(s_1)=p_S(s_2)=p_S(s_3)...=p_S(s_N)$, $\mathcal{H}(S)$ reaches its maximum, $log(N)$*.  
  
Here, $N$ denotes the total number of the samples from $S$. With this conclusion, we can calculate the representation capability of the binary spike feature map and the realvalued membrane potential map. Let $F_B \in \mathbb{B}^{C×H×W}$ denote as a binary feature map and  $M_R \in \mathbb{R}^{C×H×W}$ denote as a real-valued membrane potential map. For a binary spike output $o$, it can be expressed with 1 bit, thus the number of samples from $o$ is $2$. Then, the number of samples from $F_B$ is $2^{(C×H×W)}$ and $R(F_B)=log2^{(C×H×W)}=C×H×W$. While a real-valued membrane potential needs $32$ bits, which consists of $2^{32}$ samples. Hence, $R(M_R)=log2^{32×(C×H×W)}=32×C×H×W$. **It is thus clear that the representation capability of binary spike feature map is much limited and quantizing the real-valued membrane potentials to binary spikes induces excessive information loss.** A common consensus is that increasing the timesteps of the SNN can improve the accuracy. This can also be proved by our information theory here. Increasing the timesteps is equivalent to increasing the neuron output spike bits through the temporal dimension, thus increasing the representation capability of the output feature map.  
  
### Ternery Spike Neuron Model  
  
Improve the spike neuron activation information capacity can increase task performance. And one cannot discard the spike information processing paradigm to increase representation capability, otherwise, the event-driven and addition-only-based energy efficiency will be lost. To boost the information capacity while keeping these advantages, here, we present a ternary LIF spike neuron, given by  
$$  
u^t=\tau u^{t-1} (1-|o^{t-1}|)+\sum_j w_j o^t_{j,pre} \tag{7}  
$$  
and  
$$  
o^t=  
\begin{cases}  
1, \ \text{if} u^t ≥ V_{th} \\  
-1, \ \text{if} u^t ≤ -V_{th} \\  
0, \ \text{otherwise}  
\end{cases}  
\tag{8}  
$$  
**The SNN's event-driven signal processing characteristic makes it much energy-efficient.** In specific, only if the membrane potential exceeds the firing threshold, $V_{th}$, the spiking neuron will present a signal and start subsequent computations, otherwise, it will keep silent. For the ternary spike neuron, it enjoys the event-driven characteristic too. Rather, only if the membrane potential is greater than $V_{th}$ or less than $-V_{th}$, the ternary spike neuron will be activated to fire the $1$ or $-1$ spikes. Multiplication-addition transform is another advantage of SNNs to keep energy-efficient. **In binary spike neuron, when a spike is fired, it will be multiplied by a weight connected to another neuron to transmit information**, which can be expressed as  
$$  
x=1×w \tag{9}  
$$  
Since the spike amplitude is $1$, the multiplication can be replaced by an addition operation as  
$$  
x=0+w \tag{10}  
$$  
For a ternary spike neuron, since the spike is $1$ or $-1$, the multiplication will be  
$$  
x=1×w, \ or, \ -1×w \tag{11}  
$$  
It can be replaced by an addition operation too as  
$$  
x=0+w, \ or, \ 0-w \tag{12}  
$$  
To conclude, the proposed ternary spike neuron will enhance the expression ability of the SNN, at the same time, will retain the event-driven and addition-only advantages as the vanilla SNN too.  
  
### Trainable Ternary Spike  
  
Another problem as we aforementioned is that most prior work quantizes the membrane potentials to the same spike values. However, in the paper, we find that the membrane potential distributions along layers vary greatly.  
  
![Pasted image 20250315162929.png](../static/images/Pasted%20image%2020250315162929.png)  
  
It can be seen that the distributions are very different along layers, thus quantizing the different layer’s membrane potentials to the same spike values is unreasonable. Consequently, **we advocate that different layers' membrane potential should be quantized to different spike values**. Then, we present the trainable ternary spike neuron, where the firing spike amplitude can be learned in the training phase given by  
$$  
u^t=\tau u^{t-1}(1-|b^{t-1}|)+\sum_j w_j o^t_{j,pre} \tag{13}  
$$  
and  
$$  
o^t=  
\begin{cases}  
1 \cdot a, \ \text{if} u^t ≥ V_{th} \\  
-1 \cdot a, \ \text{if} u^t ≤ -V_{th} \\  
0 \cdot a, \ \text{otherwise}  
\end{cases}  
\tag{14}  
$$  
where $a$ is a trainable factor, $b \in \{-1,0,1\}$, and $o^t=a \cdot b^t$.  With the learnable $a$, the proposed neuron can find a better spike amplitude and treat different layers' firing activity with different strategies. The trainable factor is set in a layer-wise manner in our SNN models, i.e., $a \in \mathbb{R}^{1×1×1}$.  
  
We use the trainable factor to suit the difference of membrane potential distributions in the paper, while the trainable factor is used in (Guo et al. 2022d) to learn unshared convolution kernels.  
  
However, a new problem will be introduced by the trainable spike that the multiplication of weight and activation cannot be transformed to addition and the advantage of computation efficiency of SNNs will be lost. To deal with the problem, we follow a training-inference decoupled technique [(Guo et al. 2022d)]([Real Spike：Learning Real-Valued Spikes for Spiking Neural Networks](./Real%20Spike%EF%BC%9ALearning%20Real-Valued%20Spikes%20for%20Spiking%20Neural%20Networks.md)), which can convert the different amplitude spikes into the same normalized spike in the inference phase by a re-parameterization technique, thus these advantages of the normalized spike will be retained still.  
  
**Re-parameterization technique.** For a convolution layer, we denote its input feature map and output feature map as **F** and **G** respectively. In the convolution layer, the input map will be convolved with a group of convolution kernels to form the output feature map, which can be written as  
$$  
G=K*F \tag{15}  
$$  
where $K$ is the convolutional kernel tensor and $(*)$ is the convolution operation.  
  
For standard ternary SNNs, the input map consists of normalized ternary spikes. While in trainable ternary SNNs, the SNN is trained with real-valued spikes for the purpose of measuring the difference of membrane potential distributions.  
  
In this case, the input feature map can be denoted as below according to Eq. 14  
$$  
F=a \cdot B \tag{16}  
$$  
**In inference, we can extract a part of the values from F and fold them into the convolution kernels for the trained SNN. Then a new SNN that can emit normalized ternary spikes will be obtained without changing the values of the output maps.** This process can be illustrated as follows:  
$$  
G=K*(a*B)=(a \cdot K)*B=\tilde K * B \tag{17}  
$$  
where $\tilde K$ is the transformed convolution kernel tensor. **The reparameterization provides us with a solution to convert a trainable ternary spike-based SNN into an output-invariant normalized ternary spike-based SNN by decoupling the trainingtime SNN and inference-time SNN.**  
  
![Pasted image 20250315164854.png](../static/images/Pasted%20image%2020250315164854.png)  
  
![Pasted image 20250315164916.png](../static/images/Pasted%20image%2020250315164916.png)  
  
![Pasted image 20250315164929.png](../static/images/Pasted%20image%2020250315164929.png)  
  
![Pasted image 20250315164945.png](../static/images/Pasted%20image%2020250315164945.png)