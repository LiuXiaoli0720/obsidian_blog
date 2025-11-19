## Introduction  
  
Working memory (WM) functioning is accomplished by the unfolding of cognitive subprocesses, i.e., stimulus encoding, maintenance, and retrieval, which unroll via information communication and integration. In neurophysiology, this translates into the temporary sychronization of long-distance osciallatroy neural activities in specific frequencies, forming network connection.  
  
These large-scale brain networks have first been uncovered by functional magnetic resonance imaging (fMRI) studies. In particular, the distinctive fMRI WM networks are the frontoparietal and the default mode network (DMN). **Despite the great spatial resolution, fMRI only indirectly captures slow (a few seconds) fluctuations of neuronal activity. Instead, electro- and magnetoencephalography (EEG and MEG) directly detect neural activity with milliseconds (ms) temporal resolution—the timescale of cognitive processing steps.**  
  
Recent M/EEG studies have started exploring long-distance frequency-specific or cross-frequency neural synchronizations underpinning WM. For example, theta-gamma phase-phase coupling was associated with input-template matching. On the other hand, **frequency-specific (theta, alpha, etc.) phase-coupling was shown to differentiate different WM load conditions**.  
  
**Aside from the neurophysiological description of WM, neuropsychology has conceptualized WM as a dynamic multicomponent system, in which the communicating compartments —a central executive (attention control and encoding) and two slave storage units, the phonological loop and the visuospatial sketchpad—operate the different WM processes. Whereas neuropsychology provides an integrated picture of WM, neuroimaging research has explored the spatial, temporal, and spectral dimensions in a rather scattered way.**  
  
The **time delay embedd-hidden Markov model (TDE-HMMs)** represents a potential alternative to investigate the WM network dynamics at 360°. **This technique describes the experimental data as resulting from the alternating activation of hidden states**. One can understand the parallelism with neurophysiology: the recorded brain activity results from the recurring activation of brain networks (states) that we cannot directly observe. **This method infers in an unsupervised manner a predefined number of states that depict power covariations and phasecoupling across regions throughout the data**. Therefore, the HMM states constitute spectrally defined functional networks that wax and wane over a timescale dictated by the experimental data.  
  
## Methods  
### Data acquisition  
  
**Task design.** All participants performed a **visual-verbal n-back** task during the MEG recording. **This paradigm consists of showing a sequence of letters, and the subject is instructed to respond to a target letter by pressing a button with the right hand. In the 0-back condition, the letter $\text{X}$ is the target; during the 1 and 2-back conditions, the target is any letter that coincides with the nth (n = 1,2) preceding one. The rest of the shown letters are considered distractors**.  
  
![Pasted image 20251002154629.png](../static/images/Pasted%20image%2020251002154629.png)  
  
Twelve blocks of 20 letters (stimuli) each were presented pseudo-randomly, four for each paradigm condition. The total number of target trials is 25, 23, and 28, for the 0, 1, and 2-back conditions, respectively.  
  
### MEG data preprocessing  
  
**Data preprocessing.** The pipeline uses Oxford’s Software Library and builds upon ==SPM12== and ==Fieldtrip==. **First, we coregistered the MEG data to the T1 MR image of the same subject, applying the RHINO algorithm. Here, we used the subject-specific fiducial points acquired with the Polhemus tracker, to minimize the coregistration error. Next, we downsampled the MEG data to 250 Hz and applied a band-pass filter $[1, 45]$ Hz—Butterworth IIR filter of order 5 with zero-phase forward and reverse filter, the instability is solved by reducing the order of the filter—to discard the high and low-frequency noise. We also included a notch filter at 50 Hz to remove the remaining power line effect, which could represent a source of noise for the HMM inference. We then performed artefact rejections. First, data segments of one second with an outlier standard deviation were discarded. Next, we applied the AFRICA algorithm that decomposes the data into 62 independent components (ICA) and removes those that correlate with ECG and/or EOG ($r$ > 0.5). In the last step, we again visually examined the data to verify that all major artefacts were removed.**  
  
We applied the **linearly constrained minimum variance (LCMV)** beamforming algorithm to project the sensor data onto the source space; the source reconstruction was based on a single-shell forward model in MNI space with a projection on a 5 mm dipole grid.  
  
**Parcellation.** The time course for each region of interest (ROI) was extracted as the first principal component across the voxels’ time series. As beamforming may lead to signal leakage between regions, we orthogonalized the parcels’ time series by multivariate symmetric leakage correction.  
  
**Sign-flipping.** After beamforming, the sign of the dipoles is arbitrarily assigned, and this hinders the analysis across subjects and across brain regions. Therefore, we applied the **sign-flipping algorithm**.  
  
### Time-delay embedded-hidden Markov model (TDE-HMM)  
  
HMM: [隐马尔可夫模型（HMM）](./%E9%9A%90%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E6%A8%A1%E5%9E%8B%EF%BC%88HMM%EF%BC%89.md)  
  
**Hidden Markov model.** The hidden Markov models (HMMs) assume that the observed data result from the alternating activation of a discrete number of hidden states, which in this application represent large-scale brain functional networks. The Markovian constraint stipulates that the state activated at time t depends only on the observed data at time t and the hidden state at time t-1. Each state is defined by a specific set of distribution coefficients (mean activation and covariance matrix), computed via Bayesian inference.  
  
![Pasted image 20251003193239.png](../static/images/Pasted%20image%2020251003193239.png)  
  
The model requires to set apriori the number of states to infer. We ran multiple inferences with 4, 6, 8, and 12 states to assess the model's behavior and the replicability of the results.  
  
**Time delay embedded -- HMM.** If we considered an observation MAR model with an autoregressive order $r>0$, we would realistically capture the historical temporal interactions between time series of different brain regions. The latter is not instantaneous but delayed because of the finite conduction speed in the communication between neural populations. However, the hugh number of parameters to compute ($\text{r*ROIs*ROIs}$) is computationally expensive and could lead to overfitting.  
  
**We considered 15 lagged versions of the original data matrix with $[-7:+7]$ lagged points, corresponding to a time window of 60 ms. These matrices are piled onto the original concatenated matrix, assembling an embedded matrix that is then reduced by applying principal component analysis (PCA) and extracting the 84 ($\text{2*ROIs}$) principal components.** This reduced matrix is computationally lighter and includes, indirectly, information on the historical interactions between brain regions.  
  
**Posterior probabilities -- states time courses.** Once the states are uncovered (via stochastic Bayesian inference), the following step consists of extracting the statewise posterior probabilities—the timepoint by timepoint probability that a certain state is activated given the related observed data. These constitute the states’ time courses and are computed through the forward and backward algorithm.  
  
### Temporal dimension--statistics  
  
![Pasted image 20251003201717.png](../static/images/Pasted%20image%2020251003201717.png)  
  
Figure 1a reports the event-related field analysis. The time course of each state is epoched with respect to the stimulus onset, taking an epoch of 1400 ms long, $[−200 \ 1200]$ ms. Each trial was baseline corrected considering the pre-stimulus window $[−200 \ −30]$ ms. Afterward, we ran a **two-level generalized linear model (GLM)** to investigate the task-dependent changes in the statewise activation pattern. **The GLM design matrix consisted of 7 regressors: the constant regressors (average activity overall task conditions), 0 back target, 1 back target, 2 back target, 0 back distractor, 1 back distractor, 2 back distractor—the 6 paradigm conditions. Additional contrast regressors evaluate the effect of response (target vs distractor) and working memory load (0, 1, and 2 back). The GLM first computed the contrast of parameter estimates (COPEs) for each subject (first level), and afterward, the mean COPEs per subject were fitted across subjects (second level)**.   
  
### Spectral dimension  
  
**Power spectral density and Coherence.** As described in Fig. 1b, **the concatenated original matrix is weighted by the statewise time course—the posterior probabilities. In this way, we obtain a version of the MEG data that describes the activation of each state. Next, we apply a non-parametric multitaper estimation of spectral density, for each state and subject individually, in the broad frequency band 1–40 Hz**. We obtained the **PSD**, power distribution over the brain, from which we compute the **coherence**, resulting in the **statewise connectivity matrix**. **For visualization, the PSD maps are normalized (z-score); instead, the coherence matrix is thresholded applying a Gaussian mixture model (GMM) to identify, across subjects, the strongest connections characterizing the statewise phase-coupling network**.  
  
## Results  
  
![Pasted image 20251003203203.png](../static/images/Pasted%20image%2020251003203203.png)  
  
![Pasted image 20251003203228.png](../static/images/Pasted%20image%2020251003203228.png)  
  
![Pasted image 20251003203252.png](../static/images/Pasted%20image%2020251003203252.png)  
  
![Pasted image 20251003203311.png](../static/images/Pasted%20image%2020251003203311.png)  
  
