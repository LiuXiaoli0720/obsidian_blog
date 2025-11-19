## Introduction  
  
Spiking neural networks (SNNs) have emerged as a promising research area, offing lower energy consumption and superior robustness compared to conventional articifial neural networks (ANNs). These characteristics make SNNs highly promising for temporal data processing and power-critical applications. In recent years, significant progress has been made by incorporating ackpropagation into SNNs, which allows the integration of various ANN modules into SNN architectures, including batch normalization blocks and residual blocks. By leveraging these ANN-based methods, it becomes possible to train large-scale SNNs while preserving the energy efficiency associated with SNN's binary spiking nature.  
  
Despite significant progress, SNNs have yet to fully rxploit the superior representational capability of deep learning, primarily due to their unique training mode, which struggles to model complex channel-temporal relationships effectively. **However, existing methods handle temporal and channel information separately, leading to limited joint information extraction.** Given that SNNs reuse network parameters at each time step, there exists untapped potential for recalibration at both temporal and channel dimensions. Especially, TA-SNN [Temporal-wise Attention SNN for Event Streams Classification](./Temporal-wise%20Attention%20SNN%20for%20Event%20Streams%20Classification.md) proposed by Yao et al.  
  
Previous studies in ANNs have often utilized the attention mechanism as a means to address the challenges posed by multidimensional dynamic problems. The attention mechanism, inspired by human cognitive processes, enables the selective focus on relevant information while disregarding irrelevant data. This approach has shown promise in the realm of SNNs and merits further exploration.  
  
In this article, we involve both temporal and channel attention mechanisms in SNNs, which is implemented by efficient 1-D convolution. Fig. 1(b) shows the whole structure, we argue that this cooperative mechanism can enhance the discrimination of the learned features and can make the temporal-channel learning in SNNs easier. The main contribution of this work can be summarized as follows.  
1) We introduce a plug-and-play block into SNNs by considering the temporal and channel attentions cooperatively, which model temporal and channel information in the same phase, achieving better adaptability and bio-interpretability. To the best of our knowledge, this is the first attempt to incorporate the temporal-channel attention mechanism into the most extensively used model, leaky-integrate-and-fire (LIF)-based SNNs.   
2) A cross-convolutional fusion (CCF) operation with a cross-receptive field is proposed to make use of the associated information. It not only uses the benefit of convolution to minimize parameters but also integrates features from both temporal and channel dimensions in an efficient fashion.   
3) Experimental results show that the temporal-channel joint attention mechanism for SNN (TCJA-SNN) outperforms previous methods on static and neuromorphic datasets for classification tasks. It also performs well in generation tasks.  
  
![Pasted image 20240823094745.png](../static/images/Pasted%20image%2020240823094745.png)  
  
## Related Works and Motivation  
  
### Training Techniques for SNNs  
  
The direct application of various ANN algorithms for training deep SNNs, including gradient-descent-based methods, has gained traction. However, the nondifferentiablity of spikes poses a significant challenge. The Heaviside function, commonly used to trigger spikes, has a derivative that is zero everywhere except at the origin, rendering gradient-based learning infeasible. To overcome this obstacle, the commonly employed solutions are ANN-to-SNN and the surrogate gradient descent method.  
  
During the forward pass, the Heaviside function is retained, while a surrogate function replaces it during the backward pass. One simple choice for the surrogate function is the Spike-Operator, which exhibits a gradient resembling a shifted ReLU function. In our work, we go beyond the conventional surrogate gradient method and introduce two additional surrogate functions: the ATan surrogate function and the triangle-like surrogate function designed by Fang et al. and Bellec et al.  
  
### Attention Mechanism in Convolutional Neural Networks  
  
The SE block can be seamlessly incorporated into a network, requiring only a minimal increase in parameters to recalibrate channel information. Employing squeezing and fully connecting operations allows the network to learn a trainable scale factor for each channel. This recalibration process significantly improves the discriminative power of individual channels.  
  
### Motivation  
  
The utilization of a temporal-wise attention mechanism in SNNs has exhibited substantial progress in effectively processing time-related data streams. Moreover, it has been observed in both bioligical neural networks and ANNs that recalibrating channel features within convolutional layers hold considerable potential for enhancing performance. **Neverless, the existing SNNs-based works only process the data with either temporal or channel dimensions, thereby constraining the capacity for joint feature extraction.**  
  
![Pasted image 20240823102310.png](../static/images/Pasted%20image%2020240823102310.png)  
  
To fully use this associated information, we propose the temporal-channel joint attention (TCJA) module, a novel approach for modeling temporal and channel-wise frame correlations. Furthermore, considering the inevitable increases in the model parameters caused by the attention mechanism, we attempt to adopt the 1-D convolution operation to gain a reasonable tradeoff between model performance and parameters. Furthermore, existing SNN attention mechanisms primarily prioritize classification tasks, neglecting the needs of generation tasks. Our goal is to introduce an attention mechanism that can proficiently handle both classification and generation tasks, thereby establishing a universal attention mechanism for SNNs.  
  
## Methodology  
  
### Leaky Integrate and Fire Model  
  
Various spiking neuron models have been proposed to simulate the functioning of biological neurons, and among them, the LIF model achieves a commendable balance between simplicity and biological plausibility. The membrane potential dynamics of an LIF neuron can be descibed as  
$$\tau \frac{dV(t)}{dt}=-(V(t)-V_{reset})+I(t)$$  
where τ denotes a time constant, V (t) represents the membrane potential of the neuron at time t, and I (t) represents the input from the presynaptic neurons. For better computational tractability, the LIF model can be described as an explicitly iterative version:  
$$  
\begin{cases}  
V^n_t=H^n_{t-1}+\frac{1}{\tau}(I^n_{t-1}-(H^n_{t-1}-V_{reset})) \\  
S^n_t=\Theta(V^N_{t}-V_{threshold}) \\  
H^n_t=V^n_t·(1-S^n_t)  
\end{cases}  
$$  
$V^n_t$ represents the membrane potential of neurons within the nth layer at time $t$. $τ$ is a time constant, $S$ is the spiking tensor with binary value, I denotes the input from the previous layer, $\Theta(·)$ denotes the Heaviside step function, and $H$ represents the reset process after spiking. In our method, the parameters of the LIF model are set as follows: $τ$ = 2, $V_{reset}$ = 0, and $V_{threshold}$ = 1.  
  
### Temporal-Channel Joint Attention  
  
We contend that the frame at the current time step exhibits a significant correlation with its neighboring frames in both the channel and temporal dimensions.   
  
Initially, we employed a fully connected (FC) layer to establish the correlation between the temporal and channel information, as it provides the most direct and prominent connection between these dimensions. However, as the number of channels and time steps increases, the number of parameters grows rapidly with a ratio of $T^2 × C^2$, as illustrated in Fig. 3. Our subsequent attempt involved utilizing a 2-D convolutional layer for building this attention mechanism. Nevertheless, this approach encountered a limitation due to the fixed kernel size, which restricts the receptive field to a confined local area. In conventional CNNs, augmenting the number of layers can expand the receptive field. However, within the context of attention mechanisms, the feasibility of layer stacking, analogous to convolutional networks, is constrained, thereby limiting the receptive field when employing 2-D convolutions. For this reason, **it is necessary to decrease the number of parameters while increasing the receptive field**.   
  
![Pasted image 20240823104548.png](../static/images/Pasted%20image%2020240823104548.png)  
  
To effectively incorporate both temporal and channel attention dimensions while minimizing parameter usage and maximizing the receptive field, we present a novel attention mechanism termed TCJA. This attention mechanism is distinguished by its global cross-receptive field and its ability to achieve effective results with relatively fewer parameters, specifically $T^2 + C^2$.  
  
![Pasted image 20240823103747.png](../static/images/Pasted%20image%2020240823103747.png)  
  
**(1) Average Matrix by Squeezing:** To efficiently capture the temporal and channel correlations between frames, we first perform the squeeze step on the spatial feature maps of the input frame stream $X \in R^{T×H×W×C}$, where $C$ denotes the channel size, and $T$ denotes the time step. The squeeze step calculates an average matrix $Z \in R^{C×T}$ and element $Z_{(c, t)}$ of the average matrix $Z$ as  
$$Z_{(c,t)}=\frac{1}{H×W}\sum^{H}_{i=1}\sum^{W}_{j=1}X^{(c,t)}_{i,j}$$  
  
**(2) Temporal-Wise Local Attention:**  Following the squeeze operation, we propose the TLA mechanism for establishing temporal-wise relationships among frames. We argue that the frame in a specific time step interacts substantially with the frames in its adjacent positions. Therefore, we adopt a 1-D convolution operation to model the local correspondence in the temporal dimension, as shown in Fig. 4. In detail, to capture the correlation of input frames at the temporal level, we perform C-channel 1-D convolution on each row of the average matrix Z, and then accumulate the feature maps obtained by convolving different rows of the average matrix Z. The whole TLA process can be described as  
$$T_{i,j}=\sum^{C}_{n=1}\sum^{K_T-1}_{m=0}W^m_{(n,i)}Z_{n,j+m}$$  
$K_T (K_T < T )$ denotes the size of the convolution kernel, which indicates the number of time steps considered for the convolution operation. The parameter $W^m_{(n,i)}$ is a learnable parameter that represents the mth parameter of the $i$th channel when performing a 1-D convolution operation with $C$ channels on the $n$th row of the input tensor $Z$. $T ∈ R^{C×T}$ is the attention score matrix after the TLA mechanism.  
  
**(3) Channel-Wise Local Attention:** As aforementioned, the frame-to-frame saliency score should not only take into account the temporal dimension but also take into consideration the information from adjacent frames along the channel dimension. To construct the correlation of different frames with their neighbors channel-wise, we propose the CLA mechanism. Similarly, as shown in Fig. 4, we perform $T$-channel 1-D convolution on each column of the matrix $Z$, and then sum the convolution results of each row. This process can be described as  
$$C_{i,j}=\sum^T_{n=1}\sum^{K_C-1}_{m=0}E^m_{n,j}Z_{(i+m,n)}$$  
$K_C (K_C < C)$ represents the size of the convolution kernel, and $E^m_{(n,i)}$ is a learnable parameter, representing the mth parameter of the ith channel when performing $T$-channel 1-D convolution on the $n$th column of $Z$. $C ∈ R^{C×T}$ is the attention score matrix after CLA mechanism.  
  
To maintain dimensional consistency between the input and output, a “same padding” technique is employed in both the TLA and CLA mechanisms. This padding strategy ensures that the output dimension matches the input dimension by adding an appropriate number of zeros to the input data. Specifically, this technique involves padding the input array with zeros on both sides, where the number of zeros added is determined based on the kernel size and the stride.  
  
**(4) Cross-Convolutional Fusion:** After TLA and CLA operations, we get the temporal (TLA matrix $T$) and channel (CLA matrix $C$) saliency scores of the unput frame and its adjacent frames, respectively. Next, to learn the correlation between temporal and channel frames in tandem, we propose a crossdomain information fusion mechanism, that is, the CCF layer. The goal of CCF is to calculate a fusion information matrix $F$, and $F(⟩, |)$ is used to measure the potential correlation between the ith channel of the jth input temporal frame and other frames. The joint relationship between frames can be obtained by performing an element-wise multiplication of $T$ and $C$ as follows:  
$$  
\begin{align}  
F_{i,j}&=\sigma(T_{i,j}·C_{i,j})\\  
&=\sigma(\sum^C_{n=1}\sum^{K_T-1}_{m=0}W^m_{n,i}Z_{n,j+m}·\sum^T_{n=1}\sum^{K_C-1}_{m=0}E^m_{(n,j)}Z_{(i+m,n)})  
\end{align}  
$$  
  
### Training Framework  
  
We integrate the TCJA module into the existing benchmark SNNs and propose the TCJA-SNN. Since the process of neuron firing is nondifferentiable, we utilize the derived ATan surrogate function $\sigma'(x)=(\alpha/(2(1+((\pi /2)\alpha x)^2)))$ and the derived triangle-like surrogate function $\epsilon'(x)=(1/ \gamma ^2)max(0, \gamma-|x-1|)$ for backpropagation, which is proposed by Fang et al. and Bellec et al. , respectively. This latter function is particularly applied in the TCJA-TET-SNN, in alignment with the default surrogate function specification for temporal efficient training (TET)-based architecture.  In our method, the spike mean-square-error (SMSE) is chosen as the loss function, which can be expressed as  
$$L=\frac{1}{T}\sum^{T-1}_{t=0}L_t=\frac{1}{T}\sum^{T-1}_{t=0}\frac{1}{E}\sum^{E-1}_{t=0}(s_{t,i}-g_{t,i})^2$$  
where T denotes the simulation time step, E is the number of labels, s represents the network output, and g represents the one-hot encoded target label. We also employ the TET loss, which can be represented as  
$$L=\frac{1}{T}·\sum^{T}_{t=1}L_{CE}[s(t),g(t)]$$  
where T is the total simulation time, $L_{CE}$ denotes the crossentropy loss, $s$ is the network output, and $g$ represents the target label. The cross-entropy loss here can be represented by  
$$L_{CE}(p,y)=-\sum^{M}_{t=1}y_{o,c}log(p_{o,c})$$  
where $M$ is the number of classes, $y_{o,c}$ is a binary indicator (0 or 1) if class label $c$ is the correct classification for observation $o$, $p_{o,c}$ is the predicted probability of observation $o$ being of class $c$.  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
