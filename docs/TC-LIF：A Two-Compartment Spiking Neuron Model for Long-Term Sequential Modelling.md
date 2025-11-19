## Introduction  
  
In practice, **single-compartment spiking neurons models** have been widely adopted for large-scale brain simulations and neuromorphic computing, instances include leaky **Integrate-and-Fire (LIF) Model, Izhikevich Model, and Adaptive Exponential Integrate-and-Fire (AdEx) Model**. These single-compartment models abstract the biological neuron as a single electrical circuit, preserving the essential neuronal dynamics of biological neurons while ignoring the complex geometrical structure of dendrites as well as their interactions with the soma. This degree of abstraction significantly reduces the modeling effort, making them more feasible to study the behavior of large-scale biological neural networks and perform complex pattern recognition tasks on neuromorphic computing systems.  
  
While single-compartment spiking neuron models have demonstrated promising results on pattern recognition tasks with limited temporal context, their ability to solve tasks that require long-term trmporal dependencies remains constrained. This issue primarily arises from the difficulty of performaing long-term **temporal credit assignment (TCA)** in SNNs. **The TCA involves the identification of input spikes that contribute to future rewards or penalties, and subsequently strengthen or weaken their respective connections.** Given the discrete and sequential nature of spikes, pinpointing the exact moments or sequences that led to a prediction error becomes challenging. This issue deteriorates for long sequences, where the influence of early spikes on later predictions is more challenging to trace.  
  
Broadly speaking, **there are two research directions have been pursued to address the TCA problem in SNNs**.  
- The first direction draws inspiration from the recent success of attention models within deep learning. These methods integrate the self-attention mechanism into SNNs to enable the direct modeling of temporal relationships between different time steps. However, the self-attention mechanism is computationally expensive to operate, and it is unable to operate in real-time. Furthermore, self-attention is not directly compatible with mainstream neuromorphic chips, therefore, it cannot take advantage of the energy efficiency offered by these chips.  
- The latter research direction primarily centers around the idea of adaptive spiking neuron models.  
  
## Methodology  
### LIF Neurons Struggle to Perform Long-term TCA  
  
The neuronal dynamics of a LIF neuron can be described by the following discretr-time formulations:  
$$  
\begin{align}  
& \mathcal{U}[t]= \beta \mathcal{U}[t-1]-\mathcal{V_{th}} \mathcal{S}[t-1]+\mathcal{I}(t) \tag{1} \\  
& \mathcal{I}[t]=\sum_{i}w_ix_i[t]+b \tag{2} \\  
& \mathcal{S}[t]=\Theta(\mathcal{U}[t]-\mathcal{V_{th}}) \tag{3}  
\end{align}  
$$  
where $\mathcal{U[t]}$ and $\mathcal{I[t]}$ represent the membrane potential and the input current of a neuron at time $t$, respectively. The term $\beta \equiv exp(-dt/ \tau_m)$ is the membrane decaying coefficient that ranged from $(0,1)$, in which $\tau_m$ is the membrane time constant and $dt$ is the simulation time step. $x_i$ is the output spike of input neuron $i$ from the previous layer, $w_i$ denotes the synaptic weightthat connects input neuron $i$, and $b$ represents the bias term. An output spike will be generated once the membrane potential $\mathcal{U[t]}$ reaches the neuronal firing threshold $V_{th}$.  
  
While the backpropagation-through-time (BPTT) algorithm demonstrates effectiveness in tasks that involve limited temporal context, it encounters limitations when confronted with tasks that necessitate long-term tenporal dependencies. **This is primarily attributed to the vanishing gradient problem, where the error gradients diminish during the backpropagation process.** To further elaborate on this issue, let us consider the training of a SNN with the following objective function:  
$$  
\mathcal{L}(\mathcal{\hat S}, \mathcal{S})=\frac{1}{N} \sum^N_{n=1} \mathcal{L}(\mathcal{\hat S_n},\mathcal{S_n}) \tag{4}  
$$  
where $N$ is the number of training samples, $\mathcal{L}$ is the loss function, $\mathcal{S_n}$ is the network output, and $\mathcal{\hat S_n}$ is the training target. Following the BPTT algorithm, the gradient with respect to the weight $w$ can be calculated as follows:  
$$  
\frac{\partial \mathcal{L}}{\partial {w}}=\sum^T_{t}\frac{\partial \mathcal{L}}{\partial S[T]}\frac{\partial S[T]}{\partial \mathcal{U}[T]}\frac{\partial \mathcal{U}[T]}{\partial \mathcal{U}[t]}\frac{\partial \mathcal{U}[t]}{\partial w} \tag{5}  
$$  
where for a LIF neuron with the membrane decaying rate of $\beta \in (0,1)$:  
$$  
\frac{\partial \mathcal{U}[T]}{\partial \mathcal{U}[t]}=\prod^T_{t=t+1}\frac{\mathcal{U}[i]}{\mathcal{U}[i-1]}=\beta^{(T-t)} \tag{6}  
$$  
It is obvious that as the time step T increases, the impact of time step t on its subsequent time step diminishes. This is because the membrane potential decay causes an exponential decay of early information. This problem becomes exacerbated when t is considerably smaller than T , and the value of Eq. (6) tends to 0, thus leading to the vanishing gradient problem.  
  
**Consequently, the existing singlecompartment neuron models, such as the LIF model, face challenges in effectively propagating gradients to significantly earlier time steps.** This poses a significant limitation in learning long-term dependencies, which motivates us to develop two-compartment neuron models that possess enhanced capabilities in facilitating long-term TCA.  
  
### A Generalized Two-Compartment Spiking Neuron  
  
The P-R pyramidal neurons are located in the CA3 region of the hippocampus, which plays an important role in memory storage and retrieval of animals. Researchers have simplified this neuron model as a twocompartment model that can simulate the interaction between somatic and dendritic compartments. **Drawing upon the structure of the P-R model, we develop a generalized two-compartment spiking neuron model that defined as the following.**  
  
![Pasted image 20250113105104.png](../static/images/Pasted%20image%2020250113105104.png)  
  
$$  
\begin{align}  
& \mathcal{U}^D[t]=\alpha_1 \mathcal{U}^D[t-1]+\beta_1 \mathcal{U}^D[t-1]+\mathcal{I}[t] \tag{7} \\  
& \mathcal{U}^S[t]=\alpha_2 \mathcal{U}^S[t-1]+\beta_2\mathcal{U}^D[t]-\mathcal{V_{th}}\mathcal{S}[t-1] \tag{8} \\  
& \mathcal{S}[t]=\Theta(\mathcal{U}^S[t]-\mathcal{V_{th}}) \tag{9}  
\end{align}  
$$  
where $\mathcal{U}^D$ and $\mathcal{U}^S$ represents the membrane potentials of the dendritic and the somatic compartments, respectively. $α_1$ and $\alpha_2$ are respective membrane potential decaying coefficients for these two compartments. Notably, **the membrane potentials of these two compartments are not updated independently**. Rather, **they are coupled with each other through the second term in Eqs. (7) and (8), in which the coupling effects are controlled by the coefficients $β_1$ and $β_2$**. The interplay between these two compartments enhances the neuronal dynamics and, if properly designed, can resolve the vanishing gradient problem.  
  
### TC-LIF Spiking Neuron Model  
  
In comparison to the generalized two-compartment neuron model, we drop the membane decaying factor $\alpha_1$ and $\alpha_2$ from both compartments. This modification aims to circumvent the rapid decay of memory that could cause unintended information loss. Moreover, to circumvent excess firing caused by persistent input accumulation, we set β1 and β2 to opposite signs. The dynamics of the proposed TC-LIF model are expressed as follows:  
$$  
\begin{align}  
& \mathcal{U}^D[t]=\mathcal{U}^D[t-1]+\beta_1 \mathcal{U}^S[t-1]+\mathcal{I}[t]-\gamma \mathcal{S}[t-1] \tag{10} \\  
& \mathcal{U}^S[t]=\mathcal{U}^S[t-1]+\beta_2 \mathcal{U}^D[t]-\mathcal{V_{th}}\mathcal{S}[t-1] \tag{11} \\  
& \mathcal{S}[t]=\Theta(\mathcal{U}^S[t]-\mathcal{V_{th}}) \tag{12}  
\end{align}  
$$  
where the coefficients $\beta_1 \equiv -\sigma(c_1)$ and $\beta_2 \equiv \sigma(c_2)$ determine the interaction between these two compartments.  Here, **the sigmoid function $\sigma(\cdot)$ is utilized to unsure two coefficients are within the range of $(-1,0)$ and $(0,1)$, and the parameter $c_1$ and $c_2$ can be automatically adjusted during the training process**. The **membrane potentials of both compartments are reset after the firing of the soma**. Notably, **the reset of the dendritic compartment is triggered by the backpropagating spike that is governed by a scaling factor $\gamma$**.    
  
**$\mathcal{U}^S$ is responsible for retaining short-term memory of dendritic inputs, which will be reset after neuron firing**. In constrast, **$\mathcal{U}^D$ serves as a long-term memory that retains the information about external inputs, which is only partially reset by the backpropagating spike from the soma**. In this way, the multi-scale temporal information is effectively preserved in TC-LIF.  
  
The primary cause of the gradient vanishing problem is attributed to the recursive computation of $\partial \mathcal{U}[T] / \partial \mathcal{U}[t]$. This problem can, however, be effectively alleviated in the proposed TC-LIF model, whose partial derivative $\partial \mathcal{U}[T] / \partial \mathcal{U}[t]$ can be calculated as follows:  
$$  
\frac{\partial \mathcal{U}[T]}{\partial \mathcal{U}[t]}=\prod^T_{j=t+1}\frac{\partial \mathcal{U}[j]}{\partial \mathcal{U}[j-1]}, \ \mathcal{U}[j]=[\mathcal{U}^D[j],\mathcal{U}^S[j]]^T \tag{13}  
$$  
where  
$$  
\frac{\partial \mathcal{U}[j]}{\partial \mathcal{U}[j-1]}=  
\begin{bmatrix}  
\frac{\partial \mathcal{U}^D[j]}{\partial \mathcal{U}^D[j-1]} & \frac{\partial \mathcal{U}^D[j]}{\partial \mathcal{U}^S[j-1]} \\ \frac{\partial \mathcal{U}^S[j]}{\mathcal{U}^D[j-1]} & \frac{\partial \mathcal{U}^S[j]}{\partial \mathcal{U}^S[j-1]}   
\end{bmatrix} =   
\begin{bmatrix}  
\beta_1 \beta_2 + 1 & \beta_1 \\ \beta_1 \beta_2^2 + 2 \beta_2 & \beta_1 \beta_2 + 1  
\end{bmatrix}  
\tag{14}  
$$  
  
In order to **quantify the severity of the vanishing gradient problem in TC-LIF**, we further **calculate the column infinite norm** as provided in Eq. (15) below.  
$$  
||\frac{\partial \mathcal{U}[j]}{\partial \mathcal{U}[j-1]}||_{∞}  
=max(\beta_1 \beta_2^2+\beta_1 \beta_2+2\beta_2+1,\beta_1\beta_2+\beta_1+1)=\beta_1\beta_2^2+\beta_1\beta_2+2\beta_2+1 \tag{15}  
$$  
**The infinite norm signifies the maximum changing rate of membrane potentials over a prolonged time period.** By employing the constrained optimization method to solve the lower bound of the infinite norm for $||\partial \mathcal{U}[j] / \partial \mathcal{U}[j-1]||_{∞}$, it can be found that this value is larger than 1. This suggests that the TC-LIF model can effectively address the vanishing gradient problem. However, **given that the value of infinite norm consistently exceeds 1, it inevitably faces the gradient explodig problem**. **The experimental studies show that this value is marginally greater than 1 for the majority of $\{ \beta_1, \beta_2 \}$ selected from the second quadrant, there by leading to stable training.**  
  
It is worth noting that the TC-LIF model can be reformulated into a single-compartment form:  
$$  
\mathcal{U}^S[t]=(1+\beta_1\beta_2) \mathcal{U}^S[t-1]+\beta_2 \mathcal{U}^D[t-1]+\beta_2\mathcal{I}[t]-(\beta_2 \gamma + \mathcal{V_{th}}) \mathcal{S}[t-1] \tag{16}  
$$  
  
## Experiments  
### Parameter Space Exploration for Generalized Two-Compartment Neurons  
  
To shed light on the effectiveness of the proposed parameter setting in the TC-LIF model, we conduct a grid search by initializing $\beta_1$ and $\beta_2$ across four different quadrants and evaluate their performance on the S-MNIST and PS-MNIST datasets.  
  
![Pasted image 20250113214157.png](../static/images/Pasted%20image%2020250113214157.png)  
  
**The result reveals that apart from the models initialized in the second quadrant, models in other regions struggle to converge.** Particularly, when we initialize $β$ in the third quadrant, it refers to the scenario where the partial derivatives are less than 1 which will lead to the gradient vanishing problem. In contrast, models with $β$ initialized in the first quadrant face the severe gradient exploding problem. Both the gradient vanishing and exploding problems impede network convergence. Although initializing $β$ in the fourth quadrant can alleviate these problems, it results in a consistent negative input (Eq. (11)) to the somatic compartment that will lead to poor temporal classification results as seen across these two tasks. Therefore, **we initialize the values of $β$ from the second quadrant, that is, $β_1 ∈ (−1, 0)$ and $β_2 ∈ (0, 1)$ for our TC-LIF model and we use it consistently for the rest of our experiments**.  
  
### Superior Temporal Classification Capability  
  
![Pasted image 20250113214340.png](../static/images/Pasted%20image%2020250113214340.png)  
  
### Effective Long-term Temporal Credit Assignment  
  
To enhance visual clarity, the gradient value $\mathcal{G}^n_j$ of neuron $n$ at time step $t$ is normilized as $\mathcal{G} / \sum^N_{i=0} \sum^T_{j=0} \mathcal{G}^i_j$.  
  
![Pasted image 20250114152706.png](../static/images/Pasted%20image%2020250114152706.png)  
  
### Rapid Training Convergence  
  
Taking benefits of the exception ability in performing long-term TCA, the proposed TC-LIF model ensures more stable learning and faster network convergence.   
  
![Pasted image 20250114153809.png](../static/images/Pasted%20image%2020250114153809.png)  
  
As illustrated in Figure 4, the solid line denotes the mean accuracy, while the shaded area encapsulates the accuracy standard deviation across four runs with different random seeds.   
  
![Pasted image 20250114154608.png](../static/images/Pasted%20image%2020250114154608.png)  
  
### High Energy Efficiency  
  
We count the accumulated (AC) and multiply-and-accumulate (MAC) operations consumed during input data processing and network update.   
  
![Pasted image 20250114154135.png](../static/images/Pasted%20image%2020250114154135.png)  
  
Compared to the LIF model, the proposed TC-LIF model incurs additional $nFr_{out}E_{AC}+nE_{MAC}$ operations due to the extra computations at the dendritic compartment.   
  
To obtain the total energy cost, we base our calculation on the 45nm CMOS process that has an estimated cost of $E_{AC}=0.9 \ pJ$ and $E_{MAC}=4.6 \ pJ$ for AC and MAC operations, respectively.    
  
## Supplement  
  
**S-MNIST:** The Sequential-MNIST (S-MNIST) dataset is derived from the original MNIST dataset, which consists of 60000 and 10000 grayscale images of handwritten digits for training and testing sets with a resolution of $28×28$ pixels. In the S-MNIST dataset, each image is converted into a vector of 784 time steps, with each pixel representing one input value at a certain time step.  
  
**PS-MNIST:** The Permuted Sequential MNIST dataset (PS-MNIST) is a variation of the Sequential MNIST dataset, in which the pixels in each image are shuffled according to a fixed random permutation. This dataset provides a more challenging task than S-MNIST, as the input sequences no longer follow the original spatial order of the images. Therefore, when learning this dataset, the model needs to capture complex, non-local, and long-term dependencies between pixels.  
  
**GSC:** The GSC version 2 is a collection of 105,829 on-second-long audio clips of 35 different spoken commands, such as “yes”, “no”, “up”, “down”, “left”, “right”, etc. These audio clips are recorded by different speakers in various environments, offering a diversity of datasets to evaluate the performance of our model.  
  
**SHD:** The Spiking Heidelberg Digits dataset is a spikebased sequence classification benchmark, consisting of spoken digits from 0 to 9 in both English and German (20 classes). The dataset contains recordings from twelve different speakers, with two of them only appearing in the test set. The train set contains 8,332 examples, and the test set consists of 2,088 examples (no validation set).  
  
**SSC:** The Spiking Speech Command dataset, another spike-based sequence classification benchmark, is derived from the Google Speech Commands version 2 dataset and contains 35 classes from a large number of speakers. The original waveforms have been converted to spike trains over 700 input channels. The dataset is divided into train, validation, and test splits, with 75,466, 9,981, and 20,382 examples, respectively.  
  
![Pasted image 20250114160311.png](../static/images/Pasted%20image%2020250114160311.png)  
  
![Pasted image 20250114160434.png](../static/images/Pasted%20image%2020250114160434.png)  
  
![Pasted image 20250114160455.png](../static/images/Pasted%20image%2020250114160455.png)  
