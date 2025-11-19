#### 当前完成情况  
  
**2025.01.04：**  
- 调研阶段  
**2025.01.10：**  
- 源代码跑通  
- 可能会用到的数据集：[OpenNeuro ds000117]([Multisubject, multimodal face processing - OpenNeuro](https://openneuro.org/datasets/ds000117/versions/1.1.0))、原文开源数据（Localized estimation of event-related neural source activity from simultaneous MEG-EEG with a recurrent neural network）、[EEG and epilepsy]([Tutorials/Epilepsy - Brainstorm](https://neuroimage.usc.edu/brainstorm/Tutorials/Epilepsy))、其他文章开源数据（Simultaneous human intracerebral stimulation and HD-EEG, ground-truth for source localization methods）  
**2025.01.11：**  
- 使用RNN得到源定位结果  
  
#### 当前存在问题  
  
**2025.01.04：**  
- 需要进一步调研，寻找可用的数据集  
- 如何设计SNN模型，TC-LIF能否直接用于该模型，或如何基于TC-LIF在脉冲神经元层面进行创新  
**2025.01.11：**  
- 如何设计应用于神经信号源定位的SNN模型  
- 进一步确定OpenNeuro ds000117数据集是否可用  
**2025.01.13**：  
- 尝试能否结合ternary和TC-LIF  
  
#### 拟解决方案  
  
**2025.01.04：**  
- 调研研究现状  
**2025.01.15**:  
- 脉冲神经元角度：尝试能否从ternary、TC-LIF和s-LIF2HH中获得启发，构建基于H-H模型的适合长序列任务的多室脉冲神经元  
**2025.01.15**:  
- 脉冲神经元角度：以TC-LIF为基础，尝试能否通过利用s-LIF2HH的结论，构建多室脉冲神经元模型  
  
#### 文献阅读  
  
- TC-LIF: A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling  
- Localized estimation of event-related neural source activity from simultaneous MEG-EEG with a recurrent neural networK  
- MEG and EEG data fusion: Simultaneous localisation of face-evoked responses  
- Electromagnetic Source Imaging With a Combination of Sparse Bayesian Learning and Deep Neural Network  
- ConvDip: A Convolutional Neural Network for Better EEG Source Imaging  
- MEG Source Localization via Deep Learning  
- Deep neural networks constrained by neural mass models improve electrophysiological source imaging of spatiotemporal brain dynamics  
- Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks  
- Network model with internal complexity bridges artificial intelligence and neuroscience