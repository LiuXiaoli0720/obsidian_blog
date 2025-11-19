#### 当前完成情况  
  
**2025.01.04：**  
- 基于差分注意力机制实现多模态情绪识别模型，在EAV数据集上的准确率为90.45%  
- 对EEG信号进行地形图的可视化分析，希望通过地形图分析阐明在进行语音任务时，与语义理解、语音识别相关的脑区产生激活  
  
**2025.02.13**  
- 当前选取数据集：EAV、DEAP、MAHNOB-HCI  
- 完成数据预处理代码  
  
**2025.02.13**  
- 完成模型和训练代码（训练代码还需要修改）  
  
**2025.02.18**  
- 代码已全部完成，模态维度完成对齐  
- 需要处理多模态标签的问题，以及设计专属EEG和video的SPS块  
  
**2025.02.21**：  
- 模型修改完毕  
- 数据集DEAP、HCI-Tagging已下载  
  
**2025.04.12：**  
- ACM MM-2025 投稿  
  
#### 当前存在问题  
  
**2025.01.04：**  
- 对比 EEG-文本信息&语音-文本信息 和 EEG&语音 的分类准确率  
- 如何证明文本信息确实从EEG和语音中被去除  
- 如何利用视频信息构建多模态情绪识别模型  
**2025.02.21**：  
- 模型需要调通  
- DEAP数据集中面部视频不自带标签，需要研究使用FER+MTCNN进行标注【参考 Multimodal Emotion Recognition: Emotion Classification Through the Integration of EEG and Facial Expressions】  
  
#### 拟解决方案  
  
**2025.01.04：**  
- 阅读 Disentangling Voice and Content with Self-Supervision for Speaker Recognition（RecXi）和 MTSA-SNN: A Multi-modal Time Series Analysis Model Based on Spiking Neural Network，学习第一篇文章的思路，尝试设计将文本信息从EEG和语音中剥离的模块，学习第二篇文章的思路设计模型架构  
- 对EEG信号进行图结构转换（参考LGGNet等），将和语音任务相关的电极重新排列，对语音信号则参考RecXi的处理方式  
  
**2025.01.10：**  
- 尝试能否建立Fourier-differential-attention机制（FourierFormer，NeurIPS-2022）  
- 尝试能否将Fourier-differential-attention机制融合到spike-driven transformer中，处理语言建模和图像任务  
  
**2025.02.11：**  
- 尝试结合cross-attention机制构建SNN-based多模态融合情绪识别模型  
  
#### 文献阅读  
  
- Disentangling Voice and Content with Self-Supervision for Speaker Recognition (RecXi)   
- MTSA-SNN: A Multi-modal Time Series Analysis Model Based on Spiking Neural Network  
- FourierFormer: Transformer Meets Generalized Fourier Integral Theorem  
- Fourier or Wavelet bases as counterpart self-attention in spikformer for efficient visual classification  
- CrossViT：Cross-Attention Multi-Scale Vision Transformer for Image Classification  
- Joint Multimodal Transformer for Emotion Recognition in the Wild  
- BIOT：Biosignal Transformer for Cross-data Learning in the Wild  
- Transformer-Based Spiking Neural Networks for Multimodal Audiovisual Classification  
- STDP-Based Unsupervised Multimodal Learning With Cross-Modal Processing in Spiking Neural Network  
- Event-based Multimodal Spiking Neural Network With Attention Mechanism  
- Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning