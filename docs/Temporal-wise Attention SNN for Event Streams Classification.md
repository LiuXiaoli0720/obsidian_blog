## Introduction  
  
Dynamic vision sensors (DVS) pose a new paradigm shift by using sparse and asynchronous events to represent visual information. DVS cameras encode the time, location, and polarity of the brightness changes for each pixel at an extremely high event rate (1M to 1G events per second), and exhibit advantages mainly in three aspects:  
  
**Firstly**, DVS cameras require much less resource, as the events are sparse and only triggered when the intensity changes. **Secondly**, the $μs$ temporal resolution (TR) of DVS can avoid motion blur by producing high-rate events. **Thirdly**, DVS cameras have a high dynamic range (140dB vs. 60dB of conventional cameras), which makes them able to acquire information from challenging illumination conditions.  
  
![Pasted image 20240315200151.png](../static/images/Pasted%20image%2020240315200151.png)  
  
However, **the event streams recorded by DVS cameras are usually redundant in the temporal dimension, which is caused by high TR and irregular dynamic scene changes. This characteristic makes event streams almost impossible to process directly through deep neural networks (DNNs), which are based on dense computation**. Compromising on this, additional data preprocessing is required and inevitably dilutes the advantages of low-latency and power-saving of events.  
  
In this work, we propose the temporal-wise attention SNNs (TA-SNNs) by extending the attention concept to temporal-wise input to automatically filter out the irrelevant frames for the final decision.   
- Inspired by squeeze-and-excitation (SE) block, we design **the TA module** to **obtain the statistical features of events at different times, generate the attention scores and then weigh the events by the scores**.   
- At the same time, we propose a data augmentation method called ***random consecutive slice (RCS)*** to **utilize the event data**.  
- In order to keep the event-encoded data characteristics, we then **use binary attention scores at the inference stage with a threshold in the TA module, which is termed as *input attention pruning (IAP)* and obtains an unchanged or even higher accuracy with RCS**.  
  
We summarize our contributions as follows:  
- We **propose the TA-SNNs for event streams that can undertake the end-to-end training and inference tasks with low latency, low power consumption, and high performance**. To the best of our knowledge, this is the first work to introduce temporal-wise attention into SNNs.  
- We **propose the IAP method for SNNs and get similar or even better performance compared with those using full inputs**. The IAP brings a crucial power-saving significance for SNNs and other eventbased algorithms.  
- Inspired by the data augmentation method in video recognition and overlap method for event stream process, we **introduce the RCS method to make full use of the sampled data**.  
## Frame-based Representation  
  
Event steam comprises four dimensions: two spatial coordinates, the timestamp, and the polarity of the event. The polarity indicates an increase (ON) or decrease (OFF) of brightness, where ON/OFF can be represented via +1/1 values.  
  
Assume the **TR** (***TR is a crucial parameter for frame-based representation, and generally, the bigger TR is, the high SNR we could have***) of event stream is $dt'$ and the spatial resolution is $L×B$, then the spike pattern tensor $X_{t'} \in R^{L×B×2}$ is equal to events set $E_{t'}={e_{i}|e_{i}=[x_i, y_i, t', p_i]}$ at timestamp $t'$. For frame, set a new TR $dt=dt'× \beta$, and the consecutive $\beta$ spike patterns can be grouped as a set  
$$E_t={X_{t'}} \tag{1}$$  
where $t' \in [\beta × t, \beta × (t+1)-1]$ and $\beta$ is called resolution factor. Then, the frame of input layer at $t$ time $X^{t,0} \in R^{L×R×2}$ based on $dt$ can be got by  
$$X^{t,0}=q(E_t) \tag{2}$$  
where $t \in \lbrace 1,2,...,T \rbrace$ is timestep, and **aggreation function $q(·)$** could be selected, as **non-polarty aggregation**, **accumulate aggregation**,**AND logic operation aggregation, etc**. Here we choose a simple approch, which accumulates event stream with the information of event polarity.   
  
## Spiking Neural Network Models  
  
The LIF model is a trade-off between the complex dynamic characteristics of biological neurons and the simpler mathematical form. It is suitable for simulating large-scale SNNs and can be described by a differential function  
$$\tau \frac{du(t)}{dt}=-u(t)+I(t) \tag{3}$$  
where $\tau$ is a time constant, and $u(t)$ and $I(t)$ are the membrane potential of postsynaptic neuron and the input collected from presynaptic neurons, respectively.  
  
For sasy inference and training, ==a simple iterative representation of LIF model or LIAF model ==can be described as  
$$\begin{cases}  
U^{t,n}=H^{t-1,n}+g(W^n,X^{t,n-1})\\  
Z^{t,n}=f(U^{t,n}-u_{th})\\  
H^{t,n}=(e^{-\frac{dt}{\tau}}U^{t,n})\circ(1-Z^{t,n})\\  
X^{t,n}=\begin{cases} Z^{t,n} & \text{for LIF}\\  
ReLU(U^{t,n}) & for LIAF\end{cases}  
\tag{4}  
\end{cases}$$  
where $n$ and $t$ are indices of layer and timestep, $W^n$ is the synaptic weight matrix between two adjacent layers, $g(·)$ is a function stands for convolutional operatrion or fully connected operation, $f(·)$ is a Heaviside step function that satisfies $f(x)=1$ when $x≥0$, otherwise $f(x)=0$, $u_{th}$ is the membrane potential threshold, $e^{-\frac{dt}{\tau}}$ reflects the leakage factor of the membrane potential, $\circ$ is the Hadamard product, $X$ and $H$ are spatial and temporal input, respectively, $U$ is the membrane potentian, and $Z$ is the spile tensor.  
  
### LIF-SNNs  
  
As shown in Eq.4 and Fig.2, by coupling  (耦合) $X^{t,n-1}$ from the $n-1$ layer and $H^{t-1,n}$ from the $t-1$ timestep, we can get $U^{t,n}$. If $U^{t,n}$ is greater than $u_{th}$ , the neuron executes the fire mechanism, which outputs $z^{t,n}$ as the the spatial input of next layer, i.e., $X^{t,n}=Z^{t,n}$ , and resets $U^{t,n}$ to $u_{rest}$. Meanwhile, the neuron executes the leak mechanism, and the decayed value of membrane potential $H^{t,n}$ will be used as the temporal input for the next timestep.  
![Pasted image 20240223154120.png](../static/images/Pasted%20image%2020240223154120.png)  
  
### LIAF-SNNs  
  
For the LIAF, it kees the $H^{t-1,n}$ and changes the Heaviside step function to ReLU function for $U^{t,n}$ , i.e., $X^{t,n}=ReLU(U^{t,n})$, then both spatial and temporal domains are analog values. We use the STBP and the BPTT algorithm to train LIF-SNNs and LIAF-SNNs, respectively.  
  
## Temporal-wise Attention for SNNs[Temporal-wise attention学习总结](./Temporal-wise%20attention%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93.md)  
  
The goal of TA module is to estimate the saliency of each frame. This saliency score should not only be based on the input statistical characteristic at the current timestep, but also take into consideration the information from neighboring frames. We apply the squeeze step and excitation step in temporal-wise to implement the above two points. The spatial input tensor of $n$th layer at $t$th timestep is $X^{t,n-1} \in R^{L×B×C}$ where $C$ is channel size.  
  
Squeeze step calculates a statistical vector of event numbers, and the value of statistical vector $s^{n-1} \in R^T$ at $t$th timestep is  
  
$$s^{n-1}_t=\frac{1}{L×B×C} \sum^{C}_{k=1}\sum^{L}_{i=1}\sum^{B}_{j=1}X^{t,n-1}(k,i,j) \tag{5}$$  
By executing the excitation step, $s^{n-1}$ is subjected to non-linear mapping through a two-layer fully connected network to obtain the correlation between different frames, i.e., score vector  
$$d^{n-1}=  
\begin{cases}  
\sigma (W^n_2 \delta(W^n_1(s^{n-1}))) &\text{training,}\\  
f(\sigma(W^n_2\delta(W^n_1s^{n-1}))-d_{th}) &\text{inference,}\\  
\tag{6}  
\end{cases}$$  
where $\delta$ and $\sigma$ are ReLU and sigmoid activation function, respectively, $W^n_1 \in R^{\frac{T}{r} × T}$ and $W^n_2 \in R^{T×\frac{T}{r}}$ are trainable parameter matrices, and optional parameter $\tau$ is used to control the model complexity, $f(·)$ is a Heaviside step function that is same as in Eq.4. and $d_{th}$ is the attention threshold.   
  
We use the score vector to train a complete network at the training stage. As an optional operation, at the inference stage, we discard the irrelevant frames which are lower than $d_{th}$ , and set the attention score of the other frames to 1.  
  
Finally, we use $d_{n−1}$ as the input score vector, and the final input at $t$th timestep is  
$$\widetilde X^{t,n-1}=d^{n-1}_{t}X^{t,n-1} \tag{7}$$  
where $\widetilde X^{t,n-1} \in R^{L\times R\times C}$ is $X^{t,n-1}$ with attention score at $t$th timestep in Eq.4. Then, the membrane potential behaviors of a TA-LIF and TA-LIAF layer follow  
$$U^{t,n}=H^{t-1,n}+g(W^n,\widetilde X^{t,n-1}) \tag{8}$$  
The excitation step maps the statistical vector z to a set of temporal-wise input scores. In this regard, the TA module can be deemed as a self-attention function. The main difference is that statistical vectors in the frame-based representation directly correlate with the number of events.  
  
  
  
  
