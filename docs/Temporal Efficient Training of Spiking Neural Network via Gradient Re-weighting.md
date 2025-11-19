## Introduction  
  
The output layer’s spike frequency or the average membrane potential increment is commonly used as inference indicators in SNNs. The current standard direct training (SDT) methods regard the SNN as RNN and optimize inference indicators’ distribution. They adopt surrogate gradients (SG) to relieve the non-differentiability. However, **the gradient descent with SG does not match with loss landscape in SNN and is easy to get trapped in a local minimum with low generalizability**. **Another training issue is the memory and time consumption, which increases linearly with the simulation time.**  
  
In this work, we examine the limitation of the traditional direct training approach with SG and propose the **temporal efficient training (TET)** algorithm. **Instead of directly optimizing the integrated potential, TET optimizes every moment’s pre-synaptic inputs.** As a result, it avoids the trap into local minima with low prediction error but a high second-order moment. **Furthermore, since the TET applies optimization on each time point, the network naturally has more robust time scalability.** **Based on this characteristic, we propose the time inheritance training (TIT), which reduces the training time by initializing the SNN with a smaller simulation length.** With the help of TET, the performance of SNNs has improved on both static datasets and neuromorphic datasets.   
  
![Pasted image 20240816170404.png](../static/images/Pasted%20image%2020240816170404.png)  
  
The following summarizes our main contributions:  
  
- We analyze the problem of training SNN with SG and propose the TET model, a new loss and gradient descent regime that succeeds in obtaining more generalizable SNNs.  
  
- We analyze the feasibility of TET and picture the loss landscape under both the SDT and TET setups to demonstrate TET's advantage in better geeralization.  
  
- Our sufficient experiments on both static datasets and neuromorphic datasets prove the effectiveness of the TET model. Especially on DVS-CIFAR10, we report 83.17% top-1 accuracy for the first time, which is over 10% better than the current state-of-the-art result.  
  
## Related Work  
  
**ANN-to-SNN Conversion.** Conversion approaches avoid the training problem by trading high accuracy through high latency. They convert a high-performing ANN to SNN and adjust the SNN parameters w.r.t the ANN activation value layer-by-layer.  
  
**Direct training.** In this area. SNNs are regarded as special RNNs and training with BPTT. On the backpropagation process, The non-differentiable activation term is replaced with a surrogate gradient. Compared with ANN-to-SNN conversion, direct training achieves high accuracy with few time steps but suffers more training costs. Several studies suggest that surrogate gradient (SG) is helpful to obtain high-performance SNNs on both static datasets and neuromorphic datasets.  
  
## Preliminary  
### Iterative LIF Model  
  
We adopt the Leaky Integrate-and-Fire (LIF) model and translate it to an iterative expression with the Euler method. Mathematically, the membrane potential is updated as  
$$u(t+1)=\tau u(t)+I(t) \tag{1}$$  
where $\tau$ is the constant leaky factor, $u(t)$ is the membrane potential at time $t$, and $I(t)$ denotes the pre-snaptic inputs, which is the product of synaptic weight $W$ and spiking input $x(t)$. Given a specific threshold $V_{th}$, the neuron fires a spike and $u(t)$ reset to $0$ when the $u(t)$ exceeds the threshold. So the firing function and hard reset mechanism can be descirbed as  
$$  
\begin{align}  
a(t+1) &= \Theta(u(t+1)-V_{th}) \tag{2} \\  
u(t+1) &= u(t+1)·(1-a(t+1)) \tag{3}  
\end{align}  
$$  
where $\Theta$ denotes the Heaviside step function. The output spike $a(t+1)$ will become the post synaptic spike and propagate to the next layer. In this study, we set the starting membrane $u(0)$ to $0$, the threshold $V_{th}$ to $1$, and the leaeky factor $\tau$ to $0.5$ for all experiments.  
  
The last layer’s spike frequency is typically used as the final classification index. However, adopting the LIF model on the last layer will lose information on the membrane potential and damage the performance, especially on complex tasks. Instead, we integrate the pre-synaptic inputs $I(t)$ with no decay or firing. Finally, we set the average membrane potential as the classification index and calculate the cross-entropy loss for training.  
  
### Surrogate Gradient  
  
Following the concept of direct training, we regard the SNN as RNN and calculate the gradients through spatial-temporal backpropagation (STBP):  
$$  
\frac{\partial L}{\partial W}=\sum_{t} \frac{\partial L}{\partial a(t)} \frac{\partial a(t)}{\partial u(t)} \frac{\partial u(t)}{\partial I(t)} \frac{\partial I(t)}{\partial W} \tag{4}  
$$  
where the term $\frac{\partial a(t)}{\partial u(t)}$ is the gradient of the non-differentiability step function involving the derivative of Dirac's $\delta$-function that is typically replaced by surrogate gradients with a derivable curve. So far, there are various shapes of surrogate gradients, such as rectangular, triangle, and exponential curve. In this work, we choose the surrogate gradients shaped like triangles. Mathematically, it can describe as  
$$\frac{\partial a(t)}{\partial u(t)}=\frac{1}{\gamma^2}max(0,\gamma-|u(t)-V_{th}|) \tag{5}$$  
where the $\gamma$ denotes the constraint factor that determines the sample range to activate the gradient.  
  
### Batch Normalization for SNN  
  
Batch Normalization (BN) is beneficial to accelerate training and increase performance since it can smooth the loss landscape during training. Zheng et al. (2021) modified the forward time loop form and proposed threshold-dependent Batch Normalization (tdBN) to normalize the pre-synaptic inputs $I$ in both spatial and temporal paradigms so that the BN can support spatial-temporal input. We adopt this setup with the extension of the time dimension to batch dimension. In the inference process, the BN layer will be merged into the pre-convolutional layer, thus the inference rule of SNN remain the same but with modified weight:  
$$\hat w \leftarrow W \frac{\gamma}{\alpha}, \ \hat b \leftarrow \beta + (b-\mu)\frac{\gamma}{\alpha} \tag{6}$$  
where $μ$, $α$ are the running mean and standard deviation on both spatial and temporal paradigm, $γ$, $β$ are the affine transformation parameters, and $W$, $b$ are the parameters of the pre-convolutional layer.  
  
## Methodology  
  
### Formula of Training SNN with Surrogate Gradients  
  
**Standard Direct Training.** We use $O(t)$ to represent pre-synaptic input $I(t)$ of the output layer and calculate the cross-entropy loss. The loss function of standard direct training $L_{SDT}$ is:  
$$L_{SDT}=L_{CE}(\frac{1}{T} \sum^{T}_{t=1} O(t), y) \tag{7}$$  
where $T$ is the total simulation time, $L_{CE}$ denotes the cross-entropy loss, and $y$ represents the target label. Following the chain rule, we obtain the gradient of $W$ with softmax $S(·)$ inference function :  
$$\frac{\partial L_{SDT}}{\partial W}=\frac{1}{T}\sum^T_{t=1}[S(O_{mean})-\hat y]\frac{\partial O(t)}{\partial W} \tag{8}$$  
where $O_{mean}$ denotes the average of the output $O(t)$ over time, and $\hat y$ is the one-hot coding of $y$.  
  
**Temporal Efficient Training.** We come up with a new kind of loss function $L_{TET}$ to realize temporal efficient training (TET). It constrains the output (pre-synaptic inputs) at each moment to be close to the target distribution. It is described as:  
$$L_{TET}=\frac{1}{T}·\sum^{T}_{t=1}L_{CE}[O(t),y] \tag{9}$$  
Recalculate the gradient of weights under the loss function $L_{TET}$, and we have:  
$$\frac{\partial L_{TET}}{\partial W}=\frac{1}{T}\sum^{T}_{t=1}[S(O(t))-\hat y]·\frac{\partial O(t)}{\partial W} \tag{10}$$  
### Convergence of Gradinet Descent for SDT v.s. TET  
  
In the case of SDT, the gradient consists of two parts, the error term $(S(O_{mean}) − \hat y)$ and the partial derivative of output $\frac{\partial O(t)}{\partial W}$. When the training process reaches near a local minimum, the term $(S(O_{mean}-\hat y))$ approximates $0$ for all $t=1,...,T$, ignorant of the term $\frac{\partial O(t)}{\partial W}$. For traditional ANNs, the accumulated momentum may help get out of the local minima (e.g. saddle point) that typically implies bad generalizability. However, when the SNN is trained with surrogate gradients, the accumulated momentum could be extremely small, considering the mismatch of gradients and losses. The fact that the activation function is a step one while the SG is bounded with integral constraints. This mismatch dissipates the momentum around a local minimum and stops the SDT from searching for a flatter minimum that may suggest better generalizability.  
  
In the case of TET, this issue of mismatch is relieved by reweighting the contribution of $\frac{\partial O(t)}{\partial W}$. Indeed, considering the fact that the first that the first term $(S(O(t))-\hat y)$ is impossible to be 0 at every moment of SNN since the early output accuracy on the training set is not 100%. So TET needs the second term $\frac{\partial O(t)}{\partial W}$ close to 0 to make the $L_{TET}$ convergence. This mechanism increases the norm of gradients around sharp local minima and drives the TET to search for a flat local minimum where the distribance of weight dose not cause a huge change in $O(t)$.  
  
***Lemma 4.1 $L_{SDT}$ is upper bounded by $L_{TET}$.***  
***Proof.*** Suppose $O_i(t)$ and $\hat y_i$ denote the i-th component of $O(t)$ and $\hat y$, respectively. We have:  
$$  
\begin{align}  
L_{TET} &= -\frac{1}{T}\sum^{T}_{t=1}\sum^{n}_{i=1}\hat y_{i}logS(O_i(t)) = -\frac{1}{T}\sum^n_{i=1}\hat y_i log(\prod^T_{t=1}S(O_i(t))) \\  
&= -\sum^{n}_{i=1}\hat y_i log(\prod^T_{t=1}S(O_i(t)))^{\frac{1}{T}} ≥ -\sum^{n}_{i=1} \hat y_i log(\frac{1}{T} \sum^{T}_{t=1} S(O_i(t))) \\  
& ≥ -\sum^{n}_{t=1} \hat y_i log(S(\frac{1}{T}\sum^{T}_{t=1}O_i(t)))=L_{SDT}  
\tag{11}  
\end{align}  
$$  
where the first inequality is given by the Arithmetic Mean-Geometric Mean Inequality, and the second one is given by Jensen Inequality since the softmax function is convex. As a corollary, once the LTET gets closed to zero, the original loss function LSDT also approaches zero.  
  
Furthermore, the network output O(t) at a particular time point may be a particular outlier that dramatically affects the total output since the output of the SNN has the same weight at every moment under the rule of integration. Thus it is necessary to add a regularization term like LMSE loss to confine each moment’s output to reduce the risk of outliers:  
$$L_{MSE}=\frac{1}{T}\sum^T_{t=1}MSE(O(t),\phi) \tag{12}$$  
where $\phi$ is a constant used to regulize the membrane potential distribution. And we set $\phi=V_{th}$ in our experiments. In practice, we use a hyperparameter $\lambda$ to adjust the propotion of the regular term, we have:  
$$L_{TOTAL}=(1-\lambda)L_{TET} + \lambda L_{MSE} \tag{13}$$  
It is worth noting that we only changed the loss function in the training process and did not changes SNN's inference rules in the testing phase for a fair comparision.  
  
![Pasted image 20240817152223.png](../static/images/Pasted%20image%2020240817152223.png)  
  
### Time Inheritance Training  
  
SNN demands simulation length long enough to obtain a satisfying performance, but the training time consumption will increase linearly as the simulation length grows. So how to shorten the training time is also an essential problem in the direct training field. Traditional loss function LSDT only optimizes the whole network output under a specific T , so its temporal scalability is poor. Unlike the standard training, TET algorithm optimizes each moment’s output, enabling us to extend the simulation time naturally. We introduce **Time Inheritance Training (TIT)** to alleviate the training time problem. We first use long epochs to train an SNN with a short simulation time T. Then, we increase the simulation time to the target value and retrain with short epochs. We discover that TIT performs better than training from scratch on accuracy and significantly saves the training time. Assuming that training an SNN with simulation length T = 1 cost ts time per epoch, the SNN needs 300 epochs to train from scratch, and the TIT needs 50 epochs for finetuning. So we need 1800ts time to train an SNN with T = 6 from scratch, but following the TIT pipeline with the initial T = 2 only requires 900ts. As a result, the TIT can reduce the training time cost by half.  
  
## Experiments  
  
### Model Validation and Ablbation Study  
  
**Effectiveness of TET over SDT with SG.** **We first examine whether the mismatch between SG and loss causes the convergence problem.** For this purpose, we set the simulation length to 4 and change the spike function Θ in Eqn.2 to Sigmoid σ(k · input). We find that the TET and SDT achieved similar accuracy (***Table 2***) when k = 1, 10, 20. This indicates that both TET and SDT work when the gradient and loss function match each other. **Next, we compare the results training with LSDT and LTET on SNNs (ResNet-19 on CIFAR100) training with surrogate gradient for three runs.** As shown in ***Table 1***, our proposed new TET training strategy dramatically increases the accuracy by 3.25% when the simulation time is 4 and 3.53% when the simulation time is 6. These results quantitatively support the effectiveness of TET in solving the mismatch between gradient and loss in training SNNs with SG.  
  
![Pasted image 20240817153357.png](../static/images/Pasted%20image%2020240817153357.png)  
  
**Loss Landscape around Local Minima.** **We further inspect the 2D landscapes (Li et al., 2018) of $L_{SDT}$ and $L_{TET}$ around their local minima (see *Figure. 2*) to demonstrate why TET generalizes better than SDT and how TET helps the training process jump out of the sharp local minima typically found by SDT.** First, comparing Figure. 2 A and C, we can see that although the values of local minima achieved by SDT and TET are similar in $L_{SDT}$, the local minima of TET (Figure. 2 C) is flatter than that of SDT (Figure. 2 A). This indicates that the TET is effective in finding flatter minima that are typically more generalizable even w.r.t the original loss in TET. Next, we examine the two local minima under $L_{TET}$ to see how it helps jump out the local minima found by SDT. When comparing Figure. 2 B and D, we observe that the local minima found by SDT (Figure. 2 B) is not only sharper than that found by TET (Figure. 2 D) under $L_{SDT}$ but also maintains a higher loss value. This supports our claim that TET loss cannot be easily minimized around sharp local minima (Figure. 2 B), thus preferable to converge into flatter local minima (Figure. 2 D).   
  
![Pasted image 20240817153729.png](../static/images/Pasted%20image%2020240817153729.png)  
  
**Training from SDT to TET.** In this part, we further validate the ability of TET to escape from the local minimum found by SDT. We adopt the VGGSNN with 300 epochs training on DVS-CIFAR10. First, we optimize $L_{SDT}$ for 200 epochs and then change the loss function to $L_{TET}$ after epoch 200. ***Figure 3*** demonstrates the accuracy and loss change on the test set. After 200 epochs training, SDT gets trapped into a local minimum, and the $L_{SDT}$ no longer decreases. The $L_{TET}$ is much higher than $L_{SDT}$ since SDT does not optimize it. Nevertheless, after we change the loss function to $L_{TET}$, the $L_{TET}$ and $L_{SDT}$ on the test set both have a rapid decline. This phenomenon illustrates the TET ability to help the SNN efficiently jump out of the local minimum with poor generalization and find another flatter local minimum.  
  
![Pasted image 20240817154234.png](../static/images/Pasted%20image%2020240817154234.png)  
  
**Time Scalability Robustness.** **Here, we study the time scalability robustness of SNNs trained with TET ($L_{TET}$).** First, we use 300 epochs to train a small simulation length ResNet-19 on CIFAR100 as the initial SNN. Then, we directly change the simulation length from 2 to 8 without finetuning and report the network accuracy on the test set. ***Figure 4. A*** displays the results after changing the simulation length. We use 2, 3, and 4, respectively, as the simulation length of the initial network. When we increase the simulation length, the accuracy of all networks gradually increases. After the simulation time reaches a certain value, the network performance will slightly decrease. Interestingly, SNNs trained from scratch (T=4 and T=6) are not as good as those trained following the TIT procedure.  
  
**Network Efficiency.** **In this section, we measure the relationship between energy consumption and network performance.** SNN avoids multiplication on the inference since its binary activation and event-based operation. The addition operation in SNN costs 0.9pJ energy while multiplication operation consumes 4.6pJ measured in 45nm CMOS technology (Rathi & Roy, 2020). In our SNN model, the first layer has multiplication operations, while the other layers only have addition operations. ***Figure 4. B*** summarizes the results of different simulation times. In all cases, the SNN obtained by TET has higher efficiency.  
  
![Pasted image 20240817154949.png](../static/images/Pasted%20image%2020240817154949.png)  
  
### Comparision to Exiting Works  
  
**CIFAR.** We apply TET and TIT algorithm on CIFAR (Krizhevsky et al., 2009), and report the mean and standard deviation of 3 runs under different random seeds. The λ is set to 0.05. On CIFAR10, our TET method achieves the highest accuracy above all existing approaches. Even when T = 2, there is a 1.82% increment compare to STBP-tdBN with simulation length T = 6. It is worth noting that our method is only 0.47% lower than the ANN performance. TET algorithm demonstrates a more excellent ability on CIFAR100. It has an accuracy increase greater than 3% on all report simulation lengths. In addition, when T = 6, the reported accuracy is only 0.63% lower than that of ANN. We can see that the proposed TET’s improvement is even higher on complex data like CIFAR100, where the generalizability of the model distinguishes a lot among minima with different flatness.  
  
![Pasted image 20240817155117.png](../static/images/Pasted%20image%2020240817155117.png)  
  
**ImageNet.** The training set of ImageNet (Krizhevsky et al., 2012) provides 1.28k training samples for each label. We choose the two most representative ResNet-34 to verify our algorithm on ImageNet with λ = 0.001. SEW-ResNet34 is not a typical SNN since it adopts the IF model and modifies the Residual structure. Although we only train our model for 120 epochs, the TET algorithm achieves a 1.07% increment on Spiking-ResNet-34 and a 0.96% increment on SEW-ResNet34.  
  
**DVS-CIFAR10.** The neuromorphic datasets suffer much more noise than static datasets. Thus the well-trained SNN is easier to overfit on these datasets than static datasets. DVS-CIFAR10, which provides each label with 0.9k training samples, is the most challenging mainstream neuromorphic dataset. Recent works prefer to deal with this dataset by complex architectures, which are more susceptible to overfitting and do not result in very high accuracy. Here, we adopt VGGSNN on the DVS-CIFAR10 dataset, set λ = 0.001, and report the mean and standard deviation of 3 runs under different random seeds. Along with data augmentation methods, VGGSNN can achieve an accuracy of 77.4%. Then we apply the TET method to obtain a more generalizable optima. The accuracy rises to 83.17%. Our TET method outperforms existing state-of-the-art by 11.47% accuracy. Without data augmentation methods, VGGSNN obtains 73.3% accuracy by SDT and 77.3% accuracy by TET.  
  
  
  
  
  
  
  
  
  
  
