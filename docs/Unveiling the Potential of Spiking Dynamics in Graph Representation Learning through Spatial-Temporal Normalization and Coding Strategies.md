## Introduction  
  
In recent years, there have been attempts to combine SNNs with graph-based scenarios, but these have primarily focused on applying graph theory to analyze spiking neuron characteristics and network topologies, or utilizing spiking neuron features to tackle simple graph-related problems such as shortest path, clustering, and minimal spanning tree problems.    
  
A recent study introduced graph convolution to preprocess tactile data for SNN classification, achieving high performance in sensor data classification. However, **this approach faces challenges in accommodating general graph operations and adapting to other ecenarios**.   
  
In addition, although some works have investigated the modelling of SNN and graph learning, **they overlooked a more in-depth and systematic exploration of how spiking dynamics could impact and enhance graph representation learning**.   
  
On the other hand, numerous graph neural network (GNN) models, such as Graph Convolution Network (GCN) and Graph Attention Network (GAT), have made significant advances in solving graph-related tasks. **Despite these advancements, few works have explored the interplay between graph theory and neural dynamics, and many suffer from substantial computational overhead when applied to large-scale data, potentially limiting their practical applications.**   
  
This study aims to address these challenges by investigating the potential of spiking dynamics in graph representation learning, focusing on the development of a comprehensive spike-based modeling framework. We also introduce **spatial-temporal feature normalization (STFN)** techniques to enhance training efficiency and model stability.  
  
To investigate the potential and benefits of spiking dynamics in the context of graph-structured data, we introduce an innovative graph SNN framework designed to handle non-Euclidean data using flexible aggregation methods.   
  
The development of such SNN models presents **two major challenges**:   
  
**The first is integrating multiple graph convolution operations with spiking dynamics.**   
	To address this challenge, we unfold binary node features across both temporal and spatial dimensions, proposing a comprehensive spiking message-passing technique that integrates graph filters with spiking dynamics iteratively.   
	This approach allows for the effective merging of graph  convolution operations and spiking dynamics, leveraging the unique properties of SNNs for processing complex, structured information. By iteratively incorporating graph filters into the spiking message-passing process, our framework can dynamically adapt to the intricate data structure typical of non-Eculidean spaces, enhancing the model's ability to abstract and generaliza features from graph-based input.  
  
**The second major challenge is ensuring the convergence and performance of training in graph SNNs.**  
	We propose a spatial-temporal feature normalization (STFN) algorithm. This method normalizes the instantaneous membrane potentials across both feature and temporal dimensions for each node, thereby enhancing the SNN's ability to extract latent features from aggregated signals within a graph.  
  
Our contributions can be summarized as follows:  
	**We have developed a comprehensive spike-based modeling framework, termed Graph SNNs, which integrates spike propagation with feature affine transformations, effectively reconciling graph convolution operations with spiking communication mechanisms.** This framework is highly adaptable and supports a wide range of graph propagation operations. To our knowledge, this is the first framework to establish a general spike-based approach for regular graph tasks within a gradient-descent paradigm.   
	（开发了一个全面的基于脉冲的建模框架，称为Graph SNNs，将脉冲传播和特征仿射变换相结合，有效地协调了图卷积运算与脉冲通信机制）  
	We introduce the **spatial-temporal feature normalization (STFN)** algorithm, which accounts for temporal neuronal dynamics and aligns membrane potential representations with threshold levels, significantly improving convergence and performance.   
	（引进了时空特征归一化（STFN）算法，该算法考虑了时间神经动力学，并将膜电位表示与阈值水平对齐，显著提高了显著性和性能）  
	**We have instantiated our proposed framework into multiple models, including Graph Convolutional SNNs (GC-SNN) and Graph Attention SNNs (GA-SNN), and validated their performance through extensive experiments on semi-supervised node classification tasks.** Additionally, we evaluate the computational costs, demonstrating the high efficiency and potential of SNNs for enabling graphstructured applications on neuromorphic hardware.  
	（我们将提出的框架实例化到多个模型中，包括图卷积SNN（GC-SNN）和图注意力（GA-SNN），并通过在半监督节点分类任务上的大量实验验证了性能）  
  
## Related work  
### Spiking Neural Networks  
  
The LIF model can be mathematically descibed as follows:  
  
$$\tau \frac{du(t)}{dt} = -u(t)+I(t) \tag{1}$$  
In this context, $u(t)$ denotes the membrane potential of the neuron at a given time $t$, $I(t)$ signifies the external input current, and $\tau$ represents the membrane time constant, which influences the rate of potential decay.  
  
Inspired by the complex and varied connectivity observed in the brain, spiking neural networks (SNNs) can be organized into various architectures. There are three primary methods for training SNNs: **unsupervised learning** (such as spike-timing-dependent plasticity (STDP)), **conversion learning** (also known as the ANN-to-SNN conversion method), and **supervised learning** (which employs gradient descent to train multi-layer SNNs, achieving accuracy comparable to ANNs on complex datasets).  
  
### Graph Neural Networks  
  
Graphs are common data structures used  o represent complex systems, consisting of vertices (nodes) that symboliza entities and edges that indicate relationships between these nodes. Depending on the nature of these relationships, graphs can be classified into serveral categories, such as directed or undirected, weighted or unweighted, and cyclic or acyclic.  
  
Graph Neural Networks (GNNs) are specialized models designed to work with graph data. They leverage both the features of the nodes and the graph's structural information to learn  from complex graph datasets.   
  
The architecture of Graph Neural Networks (GNNs) can be broadly classified into four categories: **Convolutional Graph Neural Networks (GCNs)**, **Graph Attention Networks (GATs)**, **Graph Autoencoders (GAEs)**, and **Spatial-Temporal Graph Neural Networks (STGNNs)**.  
  
However, **all these models encounter the over-smoothing problem as the depth of a GNN increases**. During the triaining phase, with multiple layers pf message passing, the representations of node tend to become more similar to one another. This convergence can result in the loss of important information and reduce the model's ability to distinguish between different nodes. The challenge is a major obstacle in designing deep GNNs.  
  
### Normalization  
  
Normalization has been crucial in the training of deep learning models, enhancing training efficiency and mitigating issues like vanishing or exploding gradients and internal covariate shift. By scaling data with varying distributions to a consistent range, normalization also speeds up the model's convergence.  
	Batch Normalization (BN)  
	Layer Normalization (LN)  
	Group Normalization (GN)  
  
In graph neural networks (GNNs), adapting normalization techniques poses unique challenges due to non-Eculidean nature of the data and the complex relationships between entities.   
	**Layer normalization**, tailored for GNNs, help stabilize training and enhance convergence. Normalizing the adjacency matrix using the degree matrix, known as adjacency matrix normalization, is another crucial step that ensures scaleinvariant information propagation. Additionally, spectral normalization, which controls the Lipschitz constant of the network, has been applied to GNNs to prevent overfitting to specific graph structures or nodes. To address the over-smoothing issue and simultaneously increase GNN depth, Zhou et al. introduced the [**NodeNorm**]([Understanding and Resolving Performance Degradation in Deep Graph Convolutional Networks | Proceedings of the 30th ACM International Conference on Information & Knowledge Management](https://dl.acm.org/doi/abs/10.1145/3459637.3482488)) method. This method scales the hidden features of GNNs based on the standard deviation of each individual node, making the normalization effect controllable and compatible with GCNs in various scenarios.  
  
**Despite the development of these normalization techniques for GNNs, they have not been effectively applied to Graph Spiking Neural Networks.** This gap represents a significant challenge and offers a promising research opportunity.  
  
## Graph Spiking Neural Networks  
  
### Spiking Graph Convolution  
  
Given an attributed graph $G=(\nu, \varepsilon)$, where $\nu$ represents the set of nodes and $\varepsilon$ represents the set of edges, the graph attributes are typically described by an adjacency matrix $A \in R^{N×N}$ and a node feature matrix $X \in R^{N×C}=[x_1;x_2;...;x_N]$. Here, each row $x_i$ if $X$ represents the feature vector of a node, and the adjacency matrix $A$ satisfies $A_{ii}=0$, meaning there are no self-loops.   
  
If the signal of a single node is represented by a feature vector $x \in R^{C}$, the spectral convolution is defined as the multiplication of a filter $F_{\theta}=diag(\theta)$ with the node signal $x$, expressed as:  
$$F_{\theta} \star x=UF_{\theta}U^{T}x \tag{2}$$  
where $F_{\theta}$ is parameterizzed by $\theta \in R^{C}$, and $U$ is the eigenvector matrix of the Laplacian matrix $L=I-D^{\frac{-1}{2}}{A}D^{\frac{-1}{2}}=U{\Lambda}U^{T}$. Here, $U^{T}x$ can be interpreted as the graph Fourier transform of the signal $x$, and the diagonal matrix $\Lambda$, containing the eigenvalues, can be filted using the function $F_{\theta}(\Lambda)$.   
  
**To unify the flow of information within spiking dynamics, we convert the initial node signals $X$ into binary components $\{\tilde X_{0},\tilde X_{1},...\tilde X_{t},...,\tilde X_{T-1}\}$, where $T$ represents the length of thetime window.** **This approach transforms the attributed graph from its original state into a spiking representation.** We denote the spiking node embedding at time $t$ in the $n$-th layer as $\tilde H^{n}_{t}$, with $\tilde H^{0}_{t}=\tilde X_{t}$ (where the tilde indicates binary variables in spike form). **The layer-wise spiking graph convolution in the spatial-temporal domain** is then defined as:  
$$\tilde H^{n}_{t}=\Phi(G_{c}(A, \tilde H^{n-1}_{t})W^{n},\tilde H^{n}_{t-1}) \tag{3}$$  
In this context, **$G_{c}(A, \tilde H)$ represents the spiking feature propagation along the graph's topological structure**, $\Phi(·)$ denotes the non-linear dynamic process that depends on historical states $\tilde H^{n}_{t-1}$. The matrix $W^{n} \in R^{C^{n-1}×C^{n}}$ is a layer-specific trainalbe weight parameter, where $C^{n}$ indicates the output dimension of the spiking features in the $n$-th layer, with $C^{0}=C$ bring the input feature dimension.  
  
### Iterative Spiking Message Passing  
  
We use the **leaky integrate-and-fire (LIF)** model as our fundamental neuron unit. This model is computationally efficient and widely used, while also preserving a certain degree of biological realism. The dynamics of the LIF neuron can be described by:  
$$\tau \frac{dV(t)}{dt}=-(V(t)-V-{reset})+I(t) \tag{4}$$  
In this model, $V(t)$ represents the membrane potential of the neuron at time $t$. The parameter $\tau$ is the time constant, and $I(t)$ denotes the pre-synaptic input, which is the result of synaptic weights combined with the activities of pre-synaptic neurons or external stimuli. When $V(t)$ exceeds a certain threshold $V_{th}$, the neuron generates a spike and the membrane potential is set to $V_{reset}$. After emitting a spike, the neuron begins to accumulate membrane potential $V(t)$ again in the following time steps.  
  
The spiking message passing process comprises an information propagation step and an update step, both of which occur over $T$ time steps. Let $\tilde H_{t}=[\tilde h^{0}_{t};\tilde h^{1}_{t};...;\tilde h^{N-1}_{t}] \in R^{N×C}$ represent the node embeddings at time $t$, where each $\tilde h^{i}_{t}$ corresponds to the feature vector of node $i$. **For a node $v$, we provide a general formulation for its spiking message passing as**:  
$$\tilde h^{v}_{t}=U(\sum_{u \in N(v)}P(\tilde h^{v}_{t},\tilde h^{u}_{t}, e_{vu}),\tilde h^{v}_{t-1}) \tag{5}$$  
In this context, $P (·)$ denotes the spiking message aggregation from neighboring nodes, which can be implemented using various graph convolution operations, represented as $G_{c}(·)$. The function $U (·)$ signifies the state update, governed by a non-linear dynamic system. $N (v)$ represents the set of all neighbors of node $v$ in the graph $G$, and $e_{vu}$ indicates the static edge connection between nodes $v$ and $u$, which can be naturally extended to the formalism of directed multigraphs.  
  
To integrate the LIF model into the above framework, we employ the Euler method to convert the first-order differential equation into an iterative form. We introduce a decay factor $k$ to represent the term $(1-\frac{dt}{\tau})$ and express the pre-synaptic input $I$ as $\sum_{j} W^{j}G_{c}(A, \tilde H^{j}_{t+1})$. Here, graph convolution is used to implement the propagation step $P(·)$. Incorporating the scaling effect of $\frac{dt}{\tau}$ into the weight term, we derive the following formulation:  
$$V_{t+1}=kV_{t}+\sum_{j} W^{j}G_{c}(A, \tilde H^{j}_{t+1}) \tag{6}$$  
**$G_{c}(A, \tilde H^{j}_{t+1})$ represents the aggregated feature from the pre-synaptic neurons**, with the superscript $j$ indicating the index of the pre-synapse. By incorporating the firing-and-resetting mechanism and assuming $V_{reset}=0$, the update equation can be expressed as follows:  
$$\begin{align}  
V^{n,i}_{t+1} &= kV^{n,i}_{t}(1-\tilde H^{n,i}_{t})+\sum^{l(n)}_{j}W^{n,ij}G_{c}(A,\tilde H^{n-1,j}_{t+1}) \tag{7} \\  
\tilde H^{n,i}_{t+1} &= g(V^{n,i}_{t+1}-V_{th}) \tag{8}  
\end{align}$$  
Here, $n$ denotes the $n$th layer, $l(n)$ indicates the number of neurons in that layer. $W^{ij}$ represents the synaptic weight from the $j_{th}$ neuron in the pre-layer to the $i_{th}$ neuron in the post-layer. The function $g(·)$ is the Heaviside step function, which is used to model the neuron's spiking behavior.  
  
We observe that the computation order of the affine transformation and graph convolution can be interchanged when the graph convolution operation is linear (e.g., $G_{c}(A,H)=D^{\frac{-1}{2}}AD^{\frac{-1}{2}}H$). Given that the matrix $W$ is typically dense, while $\tilde H$ is sparse and binary, prioritizing the calculation of $\tilde H$ and $W$ can reduce computational overhead by converting multiplications into additions. Under these conditions, the process in equation (7) can be reformulated as follows:  
$$V^{n,i}_{t+1}=kV^{n,i}_t(1-\tilde H^{n,i}_{t})+G_{c}(A,\sum^{l(n)_{j}}W^{n,ij}\tilde H^{n-1,j}_{t+1}) \tag{9}$$  
  
### Spatial-temporal Feature Normalization  
  
Due to the inclusion of temporal dynamics and the event-driven nature of spiking binary representation, traditional normalization techniques cannot be directly applied to Spiking Neural Networks (SNNs). Additionally, the direct training of SNNs on graph tasks does not ensure convergence or optimal performance.  
  
Considering the feature map calculation step, let $S_{t} \in R^{N×C}$ represent the instantaneous membrane potential output of all neurons in a layer at time step $t$. This can be expressed as $\sum^{l(n)}_{j} W^{n,ij}G_{c}(A,\tilde H^{n-1,j}_{t})$ as shown in equation (7), or as $\sum^{l(n)}_{j} W^{n,ij} \tilde H^{n-1,j}_{t}$ as shown in equation (9). **In the STFN process, pre-synaptic inputs are normalized along the feature dimension $C$ independently for each node. Given the importance of temporal effects in transforming node features within the topology space, normalization is also applied along the temporal dimension across consecutive time steps.**  
  
Let $S^{k,v}_t$ denote the $k_{th}$ element in the feature vector of node $v$ at time $t$. The normalization of $S^{k,v}_t$ is performed as follows:  
$$  
\begin{align}  
\tilde S^{k,v}_t &= \frac{\rho V_{th} (S^{k,v}_t-E[S^v])}{\sqrt {Var[S^v]+\epsilon}} \\  
Y^{k,v}_t &= \lambda^{k,v} \tilde S^{k,v}_t+\gamma^{k,v}  
\end{align}  
\tag{10}  
$$  
$\rho$ is a hyperparameter optimized during the training process, $\epsilon$ is a small constant to prevent division by zero, $\lambda^{k,v}$ and $\gamma^{k,v}$ are two trainable parameter vectors specific to node $v$. $E[S^v]$ and $V ar[S^v]$ represent the mean and variance of the feature values of node $v$, respectively, computed across the feature and temporal dimensions. Figure 2 illustrates the calculation process of $E[S^v]$ and $Var[S^v]$, which are defined as follows:  
$$  
\begin{align}  
E[S^v] &= \frac{1}{CT} \sum^{T-1}_{T=0} \sum^{C-1}_{k=0}S^{k,v}_t \\  
Var[S^v] &= \frac{1}{CT} \sum^{T-1}_{t=0}\sum^{C-1}_{k=0}(S^{k,v}_t-E[S^v])^2 \\  
\tag{11}  
\end{align}  
$$  
![Pasted image 20240805195929.png](../static/images/Pasted%20image%2020240805195929.png)  
  
### Graph SNNs Framework  
  
Our framework comprises two phases: **spiking message passing** and **readout**. The spiking message passing phase involves iteratively passing messages for $T$ time steps during the inference process. After completing these iterations, the output spiking signals are decoded and transformed into high-level representations for downstream tasks. Under the rate coding condition, the readout phase involves computing the decoded feature vector for the entire graph using a readout function $R$, as described by the following formula:  
$$\hat y^v = R({\frac{1}{T} \sum^{T-1}_{t=0} \tilde h^v_t | v \in G}) \tag{12}$$  
The readout function operates on the set of node states and must be invariant to permutations of these states to ensure that Graph SNNs are invariant to graph isomorphism. This means the function should produce the same output for isomorphic graphs, regardless of node order.  
  
![Pasted image 20240805202711.png](../static/images/Pasted%20image%2020240805202711.png)  
  
**In the temporal domain**, the encoded binary node spikes are used as the current node feature at each time step $t$.  
  
**In the spatial domain**, the spiking features of a batch of nodes are first aggregated via edges connected to one-hop neighbors, using a propagation operation compatible with multiple proposed methods. These aggregated features are then fed into the spiking network module. **Within each network module, we apply Spatial-Temporal Feature Normalization (STFN) to normalize the instantaneous membrane potential output along both the feature and temporal dimensions.** This normalization ensures that the node features across all dimensions follow the distribution $N (0, (ρV_{th})^2)$. Each layer in the network encapsulates spatial-temporal dynamics for feature abstraction and mapping, and the stacked multi-layer structure enhances the SNNs’ representation capabilities.  
  
To illustrate our approach, we use the **semi-supervised learning task** as an example. In this scenario, a small subset of node labels is available, and we evaluate the model’s performance by calculating the **cross-entropy error** over all labeled examples.  
$$L=-\sum_{l \in y_L} \sum^{C^L-1}_{r=0} Y_{lr} ln \sigma(\hat y_{lr}) \tag{13}$$  
where $σ(·)$ denotes the softmax function, $Y_L$ represents the set of node indices with labels, and $Y_{lr}$ denotes the true labels corresponding to the rth dimension of the $C^L$ classes.  
  
To effectively train the Graph SNN models using gradient descent, **we employ the gradient substitution method in the backward pass**. This involves using a rectangular function to approximate the derivative of the spike activity, allowing for the backpropagation of gradients through the spiking network.  
  
### Coding Strategy  
  
In addition to the rate encoding scenario discussed previously, this framework also supports various other temporal encoding schemes.  
  
**Rank Order Coding assumes that biological neurons encode information based on the order of firing within a neuron ensemble.** Consider a target neuron $i$ receiving input from a presynaptic neuron set $Q_n$ in the $n$-th layer, where each neuron fires only once, with its activation denoted as $H_j$. ROC records the relative firing order of these neurons and updates the activation of the target neuron $V^{n+1,i}$ as follows:  
$$V^{n+1,i}=\sum_{j \in Q_n} r^{order(H^{n,j})} w^{n+1}_{ij} tag{14}$$  
where $r \in (0,1)$ is a given penalty constant, and order $(H^{n,j})$ represents the firing order of neuron $j$ in the presynaptic ensemble.   
  
**At the decision layer, a winner-takes-all strategy is employed, directly outputting the feature corresponding to the neuron with the fastest spike, enabling rapid decoding within short time steps.** The characteristics of Rank Order Coding are advantageous for swift recognition and inference in static graph tasks.  
  
### Model Instantiating: GC-SNN and GA-SNN  
  
To illustrate the effectiveness of our framework and normalization technique, we implement the framework into specific models by incorporating commonly used propagation methods in GNNs, such as **graph convolution aggregators** and **graph attention mechanisms**.  
  
**Graph Convolution Spiking Neural Network (GC-SNN)**:  
$$  
\tilde h^{n,i}_t = \Phi(\sum_{j \in N(i)} \frac{1}{C_{ij}} \tilde h^{n-1,j}_t W^n + b^n,\tilde h^n_{t-1}) \tag{15}  
$$  
where $N(i)$ represents the set of neighbors of node $i$, and $c_{ij}$ is the product of the square roots of the node degrees, specifically $c_{ij}=\sqrt{N(i)} \sqrt{N(j)}$. The term $b^n$ is a trainable bias parameter.  
  
**Graph Attention Spiking Neural Network (GA-SNN)**:  
$$  
\begin{align}  
\tilde h^n_t=\Phi(a^n_{ij} \sum_{j \in N(i)} \tilde h^{n-1,j}_t W^n, \tilde h^n_{t-1}) \\  
a^{n}_{ij} = \frac{exp(f_l (a^{nT}(\tilde h^{n-1,i}_t W^n || \tilde h^{n-1,j}_t W^n)))}{\sum_{k \in N(i)} exp(f_l(a^{nT}(\tilde h^{n-1,i}_t W^n || \tilde h^{n-1,k}_t W^n)))}  
\tag{16}  
\end{align}  
$$  
where $f_l(·)$ denotes the LeakyReLU activation function, $LeakyReLU(·)$. The vector $a^n$ is a learnable weight vector, and $|| \ ||$ represents the concatenation operation.  
  
### Spatial-temporal Embedding for Downstream Tasks  
  
We further specify node and graph classification tasks on multi-graph datasets to validate the effectiveness of our method. For different downstream tasks, we generate different level representations by using the output of the last GCN layer and provide a loss function for training the network parameters.  
  
#### Node classification task  
  
Since **the node features can be obtained directly from the GCN output**, **we directly input the node features into the MLP to obtain logits $y_{i, pred} ∈ R^C$ for each class**. The formula can be expressed as:  
$$y_{i,pred}=A \ ReLU(W\tilde h^{n,i}_t) \tag{17}$$  
where $A ∈ R^{d×C}$ , $W ∈ R^{d×d}$, and the $\tilde h^{n,i}_t$ represents the $i^{th}$ node’s feature of the last GCN layer n at the last time step $t$ . To train the network parameters, we use **cross entropy between logits and the ground truth labels** as the loss function.  
  
#### Graph classification task  
  
For graph classification tasks, **we take the average of node features outputted by GCN to generate a representation of the entire graph $y_G$**. The formula can be expressed as:  
$$y_G=\frac{1}{V} \sum^{V}_{t=0} \tilde h^{n,i}_t \tag{18}$$  
where **$\tilde h^{n,i}_t$ represents the $i^{th}$ node’s feature of the last GCN layer $n$ at the last time step $t$** . Then the representation of the graph is inputted into an MLP to obtain logits $y_{pred}$.  
$$y_{pred}=A\ ReLU(W_{y_G}) \tag{19}$$  
where $A ∈ R^{d×C}$ , $W ∈ R^{d×d}$. Afterward, the **cross-entropy between the logits and ground truth labels** is used as the error for training.  
  
## Experiments  
  
### Datasets and Pre-processing  
  
**Basic experiments.** We first use three standard citation network benchmark datasets_Cora, Pubmed, and Citeseer, where nodes represent paper documents and edges are (undirected) citation links.   
  
![Pasted image 20240807155638.png](../static/images/Pasted%20image%2020240807155638.png)  
  
![Pasted image 20240807155718.png](../static/images/Pasted%20image%2020240807155718.png)  
  
In Table 2, nodes represent paper dicuments and edges represent citation links. Label rate denotes the number of labels used for training, and features denote the dimension of feature vector for each node. Each document node has a class label.  
  
We model the citation links as (undirected) edges and construct a binary, symmetric adjacency matrix A. Note that node features correspond to elements of binary bag-of-words representation. We treat the binary representations as spike vectors, and re-organize them as an assembling sequence set w.r.t timing, where each component vector is considered equal at different time steps for simplicity.  
  
We reproduce GCN and GAT and implement our GC-SNN and GA-SNN models with deep graph library. The experimental settings of models with same propagation operator (graph convolution and graph attention) are kept the same in each dataset for fairness. We use Adam optimizer with an initial learning rate of 0.01 for GCN and GC-SNN, and 0.005 for GAT and GA-SNN. We use the dropout technique for avoiding over-fitting, which is set to 0.1 for GC-SNN and 0.6 for GA-SNN model. All models run for 200 epochs and are repeated 10 trials with different random seeds. In each trial, the model are initialized by a uniform initialization and trained by minimizing the cross-entropy loss on the training nodes. We use an L2 regularization with weight decay as 0.0005, and only use theavailable labels in training set and tset the accuracy using 1000 testing samples.  
  
  
  
  
  
  
  
  
