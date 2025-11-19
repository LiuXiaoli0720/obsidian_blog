## Introduction  
  
Deep artificial neural networks (ANNs) have recently made strides in emulating brain functions, they present limitations in capturing the brain's rich temporal dynamics and high energy efficiency. These limitations are alleviated by another family of neural networks, termed as spiking neural networks (SNNs), which can reach more closely the brain's capability in processing information through spatio-temporal encoding while being energy efficient.   
  
Other key advantages of SNNs lie in their **robustness and generalization**. SNNs present robust performance when generalizing across dicerse data types and conditions. Moreover, SNNs exhibit remarkable robustness in resisting adversarial attacks, as Liang et al. demonstrated that attacking SNN models needs larger perturbations than attacking ANN models.  
  
The optimization of SNNs is not easy because the functional roles of theri modelling components are quite unclear. Taking the most widely used SNN model, the **leaky integrate-and-fire (LIF)** model, as an example, it includes several modelling components such as **the membrane potential dynamics**, **the leakage of the membrane potential**, and **the spike generation mechanism**.   
  
## Spiking Neural Networks: A Brief Overview  
### LIF-based SNN Model  
  
The LIF-based spiking neuron model is a simplistic representation of  a biological neuron, which is widely used as a classic format in SNN modelling. LIF-based SNN models aim to emulate the behaviors of the brain with accurate functional emulation and high computational efficiency by leveraging the temporal dynamics of a spiking neuron.   
  
![Pasted image 20250210103905.png](../static/images/Pasted%20image%2020250210103905.png)  
  
The LIF neuron model is governed by two main parts: the membrane potential dynamics and the spike generation mechanism. The membrane potential $u(t)$ can be described by the following differential equation:  
$$  
\tau \frac{du}{dt}=-[u(t)-u_{rest}]+RI(t) \tag{1}  
$$  
where $\tau$ is the membrane time constant, $u_{rest}$ is the resting membrane potential, $R$ is the membrane resistance, and $I(t)$ represents the input current. When the membrane potential exceeds a certain threshold value $u_{th}$, the neuron would fire a spike, and the membrane potential is reset to $u_{reset}$. This process can be formulated as  
$$  
\begin{cases}  
o(t)=1 \ \& \ u(t)=u_{reset}, \ \text{if} \ u(t) ≥ u_{th} \\  
o(t)=0, \ \text{if} \ ＜ u_{th}  
\tag{2}  
\end{cases}  
$$  
where $o(t)$ represents the output spike at the $t$-th time step.  
  
The continuous LIF model can be discretizde using the Euler method, yielding the following iterative LIF model like  
$$  
\begin{cases}  
u^{t,n}_i=k_{\tau}u_i^{t-1,n}(1-o^{t-1,n}_i)+\sum_j w^n_{ij}o^{t,n-1}_{j} \\  
o^{t,n}_i=H(u^{t,n}_i-u_{th})  
\tag{3}  
\end{cases}  
$$  
where $u_i^{t,n}$ is updated at each time step in the $n$-th layer based on its previous state and the current input spikes. The synaptic weight connecting neurons $i$ and $j$ is denoted by $w^n_{ij}$, and $o^{t,n}_i$ represents the output spike of neuron $i$ at the $t$-th time step. The model also incorporates a leakage coefficient, $k_{\tau}$, to simulate the gradual decaying effect of the membrane potential over time, and a threshold potential, $u_{th}$, to control whether the neuron fires a spike. $H(\cdot)$ denotes the classic Heaviside function.  
### LIF Modelling Components  
  
***Leakage Component.*** $k_{\tau}$ represents the leakage coefficient ranging within $[0,1]$, **which determines how fast the membrane potential $u$ decays over time**. This component is responsible for the "leaky" functionality of the LIF neuron model, as it simulates the passive decay effect of the membrane potential.    
  
***Reset Component.*** The output spike $o$ reflects the firing mechanism of the membrane potential and $(1-o)$ is responsible for resetting the membrane potential once the neuron fires. When the neuron fires (i.e., $o=1$), the reset mechanism resets the membrane potential by multiplying it by zero (i.e., $(1-o)=0$) in updating the membrane potential at the next time step. Otherwise, the membrane potential will remain unchanged. **This component returns the membrane potential to a reset state after each spike event.**  
  
***Recurrence Component.*** This component is not presented in the equations above but can be added as recurrent connections at the network level. When included, the membrane potential dynamics can be rewritten as  
$$  
u^{t, n}_i=k_{\tau}u^{t-1, n}_i(1-o^{t-1, n}_i)+\sum_jv^{n}_{ij}o^{t-1, n}_{j}+\sum_jw^n_{ij}o^{t,n-1}_j \tag{4}  
$$  
where $v_{ij}^n$ represents the strength of the recurrent connection (synaptic weight) from neuron $j$ to $i$ in the $n$-th layer. $o^{t-1,n}_j$ represents the spike state of neuron $j$ at the $(t-1)$-th time step. **The recurrence component essentially provides feedback signals from the previous time step, allowing neurons to take the historic spike states of other neurons in the same layer into account when updating their membrane potentials.**    
  
## Approach  
  
![Pasted image 20250211095632.png](../static/images/Pasted%20image%2020250211095632.png)  
### Different Leakage Coefficients  
  
The leakage component determines the decaying rate of the neuron's membrane potential.  
  
***Normal leakage.*** In this case, the membrane potential decays with a normal rate:  
$$  
u^{t,n}_i=k_{\tau}u^{t-1,n}_i(1-o^{t-1,n}_i)+\sum_jw^n_{ij}o^{t,n-1}_j \tag{5}  
$$  
which represents **a balance between the cases without and complete leakage**.   
  
***Without leakage.*** In this variant, the leakage coefficient is set to one:  
$$  
u^{t,n}_i=1 \cdot u^{t-1, n}_i(1-o^{t-1,n}_i)+\sum_jw^n_{ij}o^{t,n-1}_{j} \tag{6}  
$$  
which means that the membrane potential does not decay at all. **This results in full integration of current input signals and the previous membrane potential, allowing the neuron to accumulate inputs indefinitely.**  
  
***Complete leakage.*** In this variant, the leakage coefficient is set to zero:  
$$  
u^{t,n}_i=0 \cdot u^{t-1,n}_i(1-o^{t-1,n}_i)+\sum_j w^n_{ij}o^{t,n-1}_j=\sum_j w^n_{ij}o^{t,n-1}_j \tag{7}  
$$  
which means that **the membrane potential decays completely at each single time step and is only determined by the current inputs**.  
### Different Reset Modes  
  
The reset component is responsible for resetting the membrane potential of the neuron every time it fires, which prevents the neuron from firing continuously with a large potential value.   
  
***Normal reset.*** In this case, the reset term is included in the equation:  
$$  
u^{t,n}_i=k_{\tau}u^{t-1,n}_i(1-o^{t-1,n}_i)+\sum_j w^n_{ij} o^{t,n-1}_{j} \tag{8}  
$$  
This reset term ensures that **the neuron's membrane potential will be reset to a lower state if the neuron fired in the last time step**, i.e., $o^{t-1,n}_i=1$.  
  
***Without reset.*** In this variant, the reset term is removed from the equation:  
$$  
u^{t,n}_i=k_{\tau}u^{t-1,n}_i+\sum_j w^n_{ij} o^{t,n-1}_j \tag{9}  
$$  
Consequently, the membrane potential will not be reset after it fires, allowing the neuron to continuously fire once its membrane potential reaches the threshold.   
### Different Recurrence Patterns  
  
***Without recurrence.*** In this case, there are no recurrent connections between neurons:  
$$  
u^{t,n}_i=k_{\tau}u^{t-1,n}_i(1-o^{t-1,n}_i)+\sum_j w^n_{ij}o^{t-1, n}_j \tag{10}  
$$  
***With recurrence.*** In this variant, recurrent connections between neurons are introduced into the network, as specified by the $\sum_j v^{n}_{ij} o^{t-1,n}_j$ torm:  
$$  
u^{t,n}_i=k_{\tau}u^{t-1,n}_i(1-o^{t-1,n}_i)+\sum_j w^n_{ij}o^{t-1,n}_{j}+\sum_j v^n_{ij}o^{t-1,n}_j \tag{11}  
$$  
This allows neurons to influence each other's states in a feedback loop, leading to more complex dynamics and better learning of temporal features.  
  
## Experiments  
### Accuracy Analysis on Different Benchmarks  
#### Selection of Benchmarks  
  
**Delayed Spiking XOR Problem.** The delayed spiking XOR problem is customized to test the long-term memory capabilities of different neural network models. **This problem is structured in three stages: initially, an input spike pattern with a varying firing rate is injected into the network; then, it is followed by a prolonged delay period filled with noisy spikes; finally, the network receives another spike pattern and the network is expected to output the result of an XOR operation between the initial and final input spike patterns.** The XOR operation is a concept in digital circuits, whose input and output signals only have binary states, one or zero.   
  
**Temporal Datasets.** For temporal datasets, we select two speech signal datasets, SHD and SSC, as they provide rich temporal information that can test the capability of SNNs in processing temporal dependencies.   
  
![Pasted image 20250211155055.png](../static/images/Pasted%20image%2020250211155055.png)  
  
**Spatial Datasets.** For the assessment of SNNs in handling spatial information, we select datasets whose data are distributed spatially in nature.   
  
![Pasted image 20250211162052.png](../static/images/Pasted%20image%2020250211162052.png)  
  
**Spatio-Temporal Datasets.** To bridge the gap between purely temporal and spatial datasets, we additionally select datasets collected by DVS cameras, including Neuromorphic-MNIST (N-MNIST) and DVS128 Gesture.  
  
![Pasted image 20250211162314.png](../static/images/Pasted%20image%2020250211162314.png)  
  
#### Experimental Setup  
  
![Pasted image 20250211163051.png](../static/images/Pasted%20image%2020250211163051.png)  
  
![Pasted image 20250211163112.png](../static/images/Pasted%20image%2020250211163112.png)  
  
#### Results and Analysis  
  
The influence of the three modelling components of SNNs, leakage, reset, and recurrence, varies significantly on different types of benchmarks. Notably, **the impact is most significant on the temporal benchmarks, followed by the spatio-temporal datasets, with the least influence observed on spatial datasets**.  
  
**The role of the leakage.** The leakage coefficient directly determines the decaying rate of the membrane potential over time. In processing long temporal sequences, the decaying rate of the membrane potential plays a crucial role in a model's ability to learn long-term dependencies. A higher leakage rate (i.e., smaller $k_{\tau}$) tends to make it difficult for the model to retain and utilize information over a long period.  
  
![Pasted image 20250211164435.png](../static/images/Pasted%20image%2020250211164435.png)  
  
**The role of the reset mode.** The reset mechanism of the LIF neuron model serves as another critical role in learning temporal dynamics. When the membrane potential surpasses a certain threshold, the neuron fires a spike, and the reset mechanism subsequently reinitializes the membrane potential to a lower value. **While this mechanism aligns with biological plausibility and aims to prevent unbounded potential accumulation, it inadvertently disrupts the temporal continuity that matters in performing certain tasks.**  
  
Specifically, in the delayed spiking XOR problem, the reset mechanism in the baseline model shows a significant degradation in accuracy. **The reset process erases the membrane potential completely, cleaning all historic information, which makes it failed to memorize long-term deoendencies in this task.** This effect is weakened on spatiotemporal and purely spatial benchmarks due to the poorer temporal information. Notice that the impact of the reset mechanism is smaller than that of the leakage component on SHD and SSD datasets with higher complexity, therefore the observed differences are not significant.  
  
![Pasted image 20250211165607.png](../static/images/Pasted%20image%2020250211165607.png)  
  
In addition, **the reset mechanism greatly impacts the firing rate of neurons**. As illustrated in Figure 4, **the reset can reduce the firing rate by returning the membrane potential to a lower value every time a spike fires**. **This action leads to sparser spike activities, which is advantageous for higher computational efficiency.** Such a feature is particularly beneficial in edge computing devices, where computational resources and power supply are limited.  
  
**The role of the recurrence pattern.** The incorporation of the recurrence component in SNNs endows these models with the capability to exchange information between different neurons in the same layer across time steps. This cross-neuron information fusion significantly enhances the model’s capability in learning, capturing, and integrating temporal features. **In temporal computing tasks, the presence of recurrence generally results in superior accuracy.** Similarly, the improvement would decrease when the benchmarks have fewer temporal dependencies such as on spatial benchmarks.  
  
### Generalization Analysis on Spatio-Temporal datasets  
#### Experimental Setup  
  
We process the N-MNIST dataset with varying temporal integration lengths to generate different frame-like sequence datasets. As shown in Figure 5, **this approach generates multiple datasets, each of which is characterized by a specific temporal resolution determined by the temporal integration length, thereby enabling a comprehensive assessment of the generalization capability of SNNs**.  
  
![Pasted image 20250211173828.png](../static/images/Pasted%20image%2020250211173828.png)  
  
**In our experiments, the N-MNIST event streams are integrated over different temporal integration lengths (i.e., 1ms, 2ms, 3ms, 5ms, and 10ms) to create varying temporal resolutions.** The primary training is conducted on the 3ms configuration, and the network structures and hyper-parameter settings are consistent with those mentioned in Table 4 and Table 5 for the N-MNIST dataset.  
  
![Pasted image 20250211174248.png](../static/images/Pasted%20image%2020250211174248.png)  
  
#### Comparing Loss Landscapes  
  
The concept of loss landscape flatness is an important concept for understanding the generalization capability of a neural network. **Flat regions in the loss landscape indicate where small variations of the network parameters result in minor changes of the loss value.** In contrast, **steep regions represent sensitive areas where minor parameter changes can lead to significant loss alterations**. Therefore, **a flatter minima in the loss landscape implies that noises and shifts in the data distribution will not lead to significant loss increases, ensuring more stable performance for unseen data**.  
  
![Pasted image 20250211174423.png](../static/images/Pasted%20image%2020250211174423.png)  
  
#### Features Space Analysis with t-SNE Visualization  
  
![Pasted image 20250211174525.png](../static/images/Pasted%20image%2020250211174525.png)  
  
### Robustness against Adversarial Attack  
  
![Pasted image 20250211174620.png](../static/images/Pasted%20image%2020250211174620.png)  
  
![Pasted image 20250211174645.png](../static/images/Pasted%20image%2020250211174645.png)  
  
## Optimization Suggestions  
  
**Suggestions for temporal computing tasks.** (1) **For tasks that need a longterm memory**, an elaborate leakage rate is critical. A high leakage rate (i.e., small kτ ) cannot memorize temporal information for a long time, thus a lower leakage rate is recommended. Note that if the task is quite difficult, e.g., SHD and SSC datasets, a too low leakage rate cannot track the fast-changing signals closely, which can also degrade the performance. If so, a learnable parameter of the leakage coefficient is recommended. (2) **For tasks that need continuous processing of temporal information without disruption** such as the delayed spiking XOR problem, disabling the reset mechanism could be beneficial. However, the increase of the firing rate without reset would decrease the computational efficiency. (3) **For tasks that model complex temporal dynamics**, the incorporation of recurrence can enhance the representation ability. However, this might lead to overfitting, for which considering the trade-off between accuracy and generalization/robustness is necessary.  
  
**Suggestions for generalization and robustness.** (1) **For tasks where generalization and robustness are paramount**, a higher leakage rate (i.e., smaller kτ ) can enhance the resistance to input perturbations by reducing error accumulation. (2) For these tasks, avoiding recurrence, although beneficial for temporal processing, is helpful for improving model generalization and robustness due to the increasing error propagation paths. Note that the gain of higher generalization and robustness might harm the application accuracy, which again reflects the trade-off mentioned above.  
