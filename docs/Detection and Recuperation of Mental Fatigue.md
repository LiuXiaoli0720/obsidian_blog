**实验设计：**  
- **疲劳生成**：通过在虚拟环境中进行一系列认知任务来诱导心理疲劳。这些任务包括有干扰项和无干扰项的两组任务。干扰项旨在分散参与者的注意力，诱发疲劳。   
- **放松疗法**：在认知任务后，参与者进入一个名为“Travelling Therapy”的VR环境，以减少负面情绪、心理疲劳并提高注意力。  
**数据收集**：  
- **眼动追踪**：使用眼动追踪设备测量瞳孔直径的变化，作为心理疲劳和工作负荷的生理指标。  
- **EEG记录**：使用EEG记录大脑的电活动，以监测心理疲劳和工作负荷的变化。  
**疲劳指标**：  
- **瞳孔直径**：瞳孔直径的变化与心理疲劳和工作负荷相关。疲劳增加时，瞳孔直径相对于基线测量会减小。  
- **EEG功率谱密度**：通过分析EEG中的不同频率带（如δ波、θ波、α波和β波）的变化来评估心理疲劳和工作负荷。  
- **任务参与指数**：基于EEG频率带的比率（Beta/(Alpha + Theta)）来衡量注意力、信息收集和视觉处理的分配。  
**数据分析**：  
- **模型**：使用**支持向量机（SVM）**中的径向基函数（RBF）核来训练一个分类器，该分类器可以从EEG数据中分类心理疲劳，测试集上的准确率达到95%。  
- **疲劳标签**：基于瞳孔直径的变化来分配疲劳标签，以识别工作负荷增加和疲劳的阶段。  
 **统计分析**：  
 - 使用非参数统计测试（如Wilcoxon符号秩检验）来比较实验不同阶段的心理疲劳指标。  
**结果**：  
- 实验结果显示，在完成认知任务期间，参与者的θ/α比率和瞳孔大小显著下降，表明心理疲劳的增加。  
- 通过VR放松疗法后的放松期间，瞳孔直径的变化没有显著差异，表明分配给放松的时间可能不足以观察到恢复技术的好处。  
  
From the result of the Welch calculation to retrieve power spectral density, the absolute power of theta, alpha, beta and delta at each electrode was calculated. Relative powers (W/Hz) were used to compute workload and task engagement using the formulas presented in  
  
Relative power of $θ= (power \ of \ θ)/(power \ of \ θ+ power \ of \ α+ power \ of \ β+ power \ of \ ∆)$   
Relative power of $α= (power \ of \ α)/(power \ of \ θ+ power \ of \ α+ power \ of \ β+ power \ of \ ∆)$   
Relative power of $β= (power \ of \ β)/(power \ of \ θ+ power \ of \ α+ power \ of \ β+ power \ of \ ∆)$   
Relative power of $∆ = (power \ of \ ∆)/(power \ of \ θ+ power \ of \ α+ power \ of \ β+ power \ of \ ∆)$  
  
![Pasted image 20241105172048.png](../static/images/Pasted%20image%2020241105172048.png)  
  
![Pasted image 20241105172124.png](../static/images/Pasted%20image%2020241105172124.png)  
  
![Pasted image 20241105172142.png](../static/images/Pasted%20image%2020241105172142.png)  
  
![Pasted image 20241105172206.png](../static/images/Pasted%20image%2020241105172206.png)  
  
![Pasted image 20241105172227.png](../static/images/Pasted%20image%2020241105172227.png)  
  
![Pasted image 20241105172300.png](../static/images/Pasted%20image%2020241105172300.png)  
  
![Pasted image 20241105172555.png](../static/images/Pasted%20image%2020241105172555.png)  
  
![Pasted image 20241105172624.png](../static/images/Pasted%20image%2020241105172624.png)  
  
![Pasted image 20241105172643.png](../static/images/Pasted%20image%2020241105172643.png)