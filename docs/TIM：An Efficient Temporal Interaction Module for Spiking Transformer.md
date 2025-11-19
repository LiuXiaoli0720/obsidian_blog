## Introduction  
  
Spikformer [SPIKFORMER：When SNN meets Transformer](./SPIKFORMER%EF%BC%9AWhen%20SNN%20meets%20Transformer.md) represents the first successful integration of the Transformer architecture into the SNN domain. This model innovatively designs Spiking Self Attention to implement Transformer attention. In recent studies of Spiking Transformers, many improvements have been made. Currently, there are two primary approaches to improving Spiking Transformers. The first approach involves enhancing network performance by modifying the attention mechanism (eg. Spikeformer and DISTA). The second focuses on leveraging the efficiency of SNNs to reduce the computational energy consumption of Transformers (e.g. SPikingformer and Spike-diriven Transformer).  
  
In this study, we have developed a Temporal Interaction Module that can be seamlessly integrated into the attention matrix computation of a Spiking Transformer. This module enables the adaptive amalgamation of historical and current information, thereby effectively capturing the intrinsic dynamics of neuromrophic data. Our experiments conducted across various neuromorphic datasets demonstrate that our approach achieves the state-of-the-art performance of Spiking Transformer. In summary, our contributions are as follows:  
  
- Through our analysis, we identified that a primary limitation in current Spiking Transformers is their insufficient handling of temporal information, chiefly because the attention matrix is reliant solely on information from the current moment.   
  
- To address this, we have designed a Temporal Interaction Module to adaptively utilizes information from different time steps and functions as a lightweight addition. It can be integrated with existing attention computation modules without significantly increasing computational load.   
  
- We conducted experiments on neuromorphic datasets, including CIFAR10-DVS, NCALTECH101, NCARS, UCF101-DVS, HMDB-DVS and SHD. We demonstrated the effectiveness and generalization ability of our method. Our results show that our approach sets a new benchmark in the Spiking Transformer domain, achieving better performance across these datasets.  
  
## Related Work  
  
### Spiking Transformer  
  
In the Transformer domain, Spikformer [SPIKFORMER：When SNN meets Transformer](./SPIKFORMER%EF%BC%9AWhen%20SNN%20meets%20Transformer.md) innovatively implemented an attention mechanism called Spiking Self Attention. This mechanism replaced the traditional activation function of ANNs with spiking neurons and omitted the softmax function typically required before calculating attention, opting instead for direct matrxi multiplication. DISTA enhanced the capture of spatiotemporal data by designing new neuronal connections. Spikeformer [Spikeformer：A Novel Architecture for Training High-Performance Low-Latency Spiking Neural Network](./Spikeformer%EF%BC%9AA%20Novel%20Architecture%20for%20Training%20High-Performance%20Low-Latency%20Spiking%20Neural%20Network.md) introduced separate Temporal and Spatial Multi0head Self Attention in each Transformer block to strengthen data processing capabilities. Spikingformer [Spikingformer：Spike-driven Residual Learning for Transformer-based Spiking Neural Network](./Spikingformer%EF%BC%9ASpike-driven%20Residual%20Learning%20for%20Transformer-based%20Spiking%20Neural%20Network.md) achieved greater efficiency by rearranging the positions of neurons and convolutional layers to eliminate float-integer multiplication. In contrast, the Spike-driven Transformer reduced time complexity by altering the order of attention computation and substituting multiplication with addition.   
  
However, despite a series of advancements in the Spiking Transformer domain, these methods still exhibit performance shortcomings when processing information such as neuromorphic data with strong temporal characteristics. This limitation reveals the current technology's inadequacies in capturing and processing data with complex temporal dependencies, particularly in analyzing the deep temporal dynamics and subtle changes within neuromorphic data.  
  
## Preliminaries  
  
### Spiking Self Attention Analysis  
  
The primary strength of the Transformer model resides in its innovative self-attention mechanism. This mechanism facilitates a nuanced examination of the relative importance assigned to distinct positions within a sequence, consequently enhancing the model’s capacity for sophisticated information processing. Building upon this foundational concept, the Spikformer model introduces an advancement with the Spike Self Attention (SSA) mechanism, a development that draws inspiration from Vaswani et al.’s seminal work. The intricacies of this mechanism are delineated in Eq. 5.  
$$  
\begin{cases}  
	Q[t],K[t],V[t]=LIFNode(X[t]) \\  
	A[t]=Q[t]K[t]^TV[t] \\  
\end{cases}  
\tag{5}  
$$  
Upon the amalgamation of SSA and LIF, it is deducible that the conduit of information transmission within the spiking Transformer architecture is characterized by Eq. 6:  
$$  
V[t]=V[t-1]+\frac{1}{\tau}(A[t]-(V[t-1]))=(1-\frac{1}{\tau})V[t-1]+\frac{1}{\tau}A[t] \tag{6}  
$$  
It is apparent that the membrane potantial at a given time, $V[t]$, is principally determined by its previous state $V[t-1]$ and the current input $A[t]$. The operation $Q[t]K[t]^{T}V[t]$ is chiefly responsible for facilitating the interaction of spatial information. However, the retention and axtraction of temporal information rely solely on the dynamic changes in the neuronal membrane potential. In the existing spike self-attention mechanisms, temporal information has not been adequately considered, a deficiency that significantly limits the spiking Transformer's capability in processing time-series information.  
  
## Method  
### Spikformer Backbone  
  
In our experiments employing the Temporal Interaction Module, we have chosen Spikformer as the primary framework, with its network architecture comprehensively illustrated in Fig. 1.  
  
![Pasted image 20240828203823.png](../static/images/Pasted%20image%2020240828203823.png)  
  
Given TIM’s emphasis on temporal enhancement within neuromorphic datasets, our inputs are formatted as Event Streams. We have preserved the key components of Spikformer for consistency across our experimental models. The Event Stream undergoes a transformation into requisite dimensions through the Spiking Patch Splitting (SPS) process, utilizing convolutional techniques. Within the Spiking Self Attention (SSA) module, the TIM Stream is employed for query operations to facilitate temporal enhancement. The Multi-Layer Perceptron (MLP) is structured with a bi-layered Convolution-Batch Normalization-Leaky Integrate-and-Fire (Conv-BN-LIF) framework. Finally, the Classification Head is constituted by a linear layer, aligning with the model’s output objectives.  
  
### TIM Unit  
  
The effective preservation, processing, and extraction of temporal features are crucial in handling neuromorphic data. As demonstrated in Eq. 5, traditional Spiking Transformers construct an attention matrix at each time step. However, this attention mechanism solely correlates with the current input, leading to a substantial underutilization of information from different time steps.  
  
Here, $f$ signifies an operation for extracting historical information, implemented as a one-dimensional convolution in this study. Consequently, $Q^{TIM}[t]$ comprises two components: the first represents the contribution of historical information to the current attention, while the second signifies the direct impact of the current input. The hyperparameter $\alpha$ facilitales an adaptive balance between the significance of historical and curent information. The specific illustration of TIM Unit can be found in Fig 2(a).  
  
![Pasted image 20240828204324.png](../static/images/Pasted%20image%2020240828204324.png)  
  
### TIM Stream  
  
As depicted in Fig.2(b), this module is integrated into the computational graph of the attention matrix.  
$$  
Attention[t]=A^{TIM}[t]K[t]^{T}V[t]=\alpha f(Q^{TIM}[t-1])K[t]^TV[t]+(1-\alpha)Q[t]K[t]^TV[t] \tag{7}  
$$  
Compared to the traditional SSA, our introduction of the $f$ operation, which employs a one-dimensional convolution, does not significantly increase the number of parameters. By incorporating the TIM-built attention matrix, the model is capable of not only processing information from the current moment but also of utilizing output information from past moments, thereby effectively capturing the intrinsic dynamics of time series. This enhancement significantly bolsters the model’s temporal memory capability and allows it to utilize historical information at each time step. This feature enables the model to dynamically adjust its behavior based on the characteristics of different tasks and data, thus improving computational efficiency and the generalization capacity of the model.  
  
## Experiments  
  
![Pasted image 20240828205753.png](../static/images/Pasted%20image%2020240828205753.png)  
  
![Pasted image 20240828205816.png](../static/images/Pasted%20image%2020240828205816.png)  
  
## Discussion  
### Ablation Study  
  
To thoroughly validate the effectiveness of our algorithm, we embarked on comprehensive ablation studies utilizing the CIFAR10-DVS and N-CALTECH101 datasets. As detailed in Tab. 3 and Fig. 3, our model demonstrated substantial performance enhancemantes over the established baseline models in both datasets.  
  
![Pasted image 20240828210026.png](../static/images/Pasted%20image%2020240828210026.png)  
  
![Pasted image 20240828210111.png](../static/images/Pasted%20image%2020240828210111.png)  
### Temporal Enhancement Validation  
  
To rigorously validate that the performance enhancements observed in the Spiking Transformer are indeed due to its improved temporal processing capabilities, and not merely a consequence of an increased parameter count, we undertook a meticulous adjustment of the Temporal Interaction Module’s structure. This refinement was crucial in isolating the impact of temporal capability enhancements from the effects of parameter augmentation.  
  
As detailed in Eq. (8) and (9), we specifically re-engineered the operational mechanism of the TIM Stream. This modification was aimed at changing the functional dynamics of the TIM Unit. Instead of allowing the TIM Unit to engage in interactions across multiple time steps, which could potentially confound our assessment of its temporal processing proficiency, we constrained its operation to the current time step only (which we call local TIM).  
$$  
\begin{align}  
Q[t]=LIFNode(TIM(Q[t])) \tag{8} \\  
Attention[t]=Q[t]K^TV[t] \tag{9}  
\end{align}  
$$  
As illustrated in Tab. 4, a notable experiment was conducted applying our model to the CIFAR10-DVS and N-CALTECH101 datasets. The results were quite revealing: the model’s accuracy decreased by 2.7% and 2.4% on these datasets, respectively. This phenomenon strongly indicates that the improvement in accuracy previously attributed to the Temporal Interaction Module is primarily a result of its enhanced ability to process temporal information, rather than the increased parameter count. The stability of the parameter count during the experiment reinforces this conclusion, highlighting the intrinsic value of TIM in specifically enhancing temporal data processing capabilities. The comprehensive training details are presented in Fig. 3.  
  
![Pasted image 20240828210659.png](../static/images/Pasted%20image%2020240828210659.png)  
  
We further concentrated on investigating the impact of the hyperparameter α on the performance of the Temporal Integration Module. This exploration was conducted through a series of experiments using the CIFAR10-DVS and NCALTECH101 datasets. The top-1 accuracy of TIM on both datasets with different α values are illustrated in Tab. 5. The findings from these experiments were quite revealing. We observed that regardless of the specific value, setting α to any non-zero number consistently resulted in better performance compared to the baseline scenario where α was set to zero. This pattern indicates that the introduction of temporal interaction in TIM significantly enhances its performance. More specifically, for the CIFAR10-DVS dataset, the model’s peak performance, with an accuracy of 81.6%, was achieved when α was set to 0.6. Similarly, for the N-CALTECH101 dataset, the optimal performance occurred at an α value of 0.4, reaching a top accuracy of 79.0%. These results highlight an important aspect of TIM’s functionality. While the model exhibits robustness to a range of α values, fine-tuning this hyperparameter allows for more precise optimization relative to specific datasets. The ability to adjust α effectively tailors the model to different data distributions, enhancing TIM’s versatility and efficacy in diverse neuromorphic data processing applications.  
  
### Efficiency and Generalizability Validation  
  
TIM introduces additional operations into the model, resulting in extra computational overhead. To alleviate concerns about TIM’s reduced efficiency, we reduced the training steps. We found that with 6 steps, TIM achieves the same performance as the baseline does with 10 steps, shown in Fig 4. This implies that TIM can save approximately 40% of the time, suggesting that TIM retains the efficiency of SNN.  
  
Furthermore, to validate the generalizability of TIM’s capabilities to other Spiking Transformers, we conducted experiments on the Spike-driven Transformer as well. Before computing the Spike Driven Self Attention (SDSA, shown in Eq. 10), where $\bigodot$ refers to element-wise multiplication while $\bigotimes$ denotes Hadamard product. We applied the same procedure to let Q pass through TIM, resulting in the curve shown in Fig 4.  
$$  
\begin{cases}  
	Q[t],K[t],V[t]=LIFNode(X[t]) \\  
	A[t]=LIFNode(K[t] \bigodot V[t]) \\  
	Attention[t]=A[t] \bigotimes Q[t] \\  
\end{cases}  
\tag{10}  
$$  
SDSA achieved a top-1 accuracy of 77% on CIFAR10-DVS with 10 steps, while SDSA+TIM achieved a top-1 accuracy of 78.5%. We consider the 1.5% difference to be statistically significant, indicating that TIM gains traction in SDSA, demonstrating its ability to generalize across a broader range of Spiking Transformer architectures.  
  
  
  
  
  
  
  
  
  
  
  
  
