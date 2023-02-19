# Introduction

BoundaryFace's motivation is partially  inspired by NPT-Loss [Arxiv](https://arxiv.org/ftp/arxiv/papers/2103/2103.03503.pdf). Due to the different research's line and the fact that BoundaryFace is not based on the innovation of NPT-Loss,  we do not consider to cite NPT-Loss and compare it with BoundaryFace in this paper.

The differences with NPT-Loss are：

- Solving different problems for face recognition:  NPT-Loss addresses the shortcomings of metric loss in face recognition, such as combinatorial explosion; and has the effect of implicitly hard-negative mining. BoundaryFace solves the shortcomings of margin-based softmax in face recognition; BoundaryFace focus on hard samples directly and can tolerate closed-set noise simultaneously.
- The motivation (idea)  is different: NPT-Loss compresses the distance between the sample and ground truth class center while increasing the distance between the sample and the nearest negative class center by using a form of proxy triplet. BoundaryFace first considers closed-set noise.  Starting from the perspective of decision boundary, based on the premise of closed-set noise label correction, the framework directly emphasizes hard sample features that are located in the margin region.  Obviously, the NPT-Loss is still heavily disturbed by noise samples.
- The final loss functions are different: NPT-Loss is an proxy-triplet's form with hyper-parameter free. BoundaryFace is the margin-based softmax form, and still has hyper-parameters.

Now, for academic rigor, we have added related work to the original paper.  See link at  [Arxiv](https://arxiv.org/pdf/2210.04567.pdf)

# Quick Start

> This repository will help you learn more about the details of our experiments in the paper.

## Training environment

Experiment based on WebFace (in paper):

|     OS     |   GPU    | Python | CUDA | torch | torchvision |
| :--------: | :------: | :----: | :--: | :---: | :---------: |
| Windows 10 | 1 TitanX | 3.7.0  | 9.0  | 1.1.0 |    0.3.0    |

packages: `requirements_ct.txt`



Experiment based on MS1M / MS1MV2:

|  OS   |     GPU      | Python | CUDA | pytorch | torchvision |
| :---: | :----------: | :----: | :--: | :-----: | :---------: |
| Linux | 8 * RTX 3070 | 3.7.0  | 11.3 | 1.10.0  |   0.11.0    |

packages: `requirements_dt.txt`

## Dataset

**For Training Set:**   [InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)

- CASIA-WebFace
- MS1M [addr](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)
- MS1MV2

**For Testing Set:** [InsightFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_):

- LFW
- AgeDB-30
- CFP-FP
- CALFW
- CPLFW
- SLLFW
- RFW [Wang et al.](http://www.whdeng.cn/RFW/testing.html)
- MegaFace



## Train

> Please modify the relevant path parameters by yourself in advance. (such as the save path of closed-set noise etc.)



**For training CASIA-WebFace and noisy synthetic datasets, we use 1 NVIDIA TitanX GPU with batch size of 64. **

- noisy synthetic datasets is made by `utils/generate_label_flip.py` and `utils/generate_outlier_update.py`
  - Artificially synthesized datasets: The same proportion of noise, due to  the different selection of noise samples and differences in the distribution of noise samples may lead to some differences in the test results with the paper, **but does not affect the conclusions in paper and README.md**.

```shell
visdom
python ./training_mode/conventional_training/train.py
```





**For training MS1M / MS1MV2, we use  8 * 3070 GPU with batch size of  8 * 32 **

|    Dataset    | backbone | total epoch | milestone | epoch_start |  m   |  s   |
| :-----------: | :------: | :---------: | :-------: | :---------: | :--: | :--: |
| MS1M / MS1MV2 | Res50-IR |     24      | 10,18,22  |     10      | 0.5  |  32  |



```shell
tensorboard --logdir ./training_mode/distributed_training/MS1M_tensorboard
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1324 ./training_mode/distributed_training/train_DDP.py
```





## Results of retraining

The following problems existed with the experiments in the paper, so we decided to retrain the experiments in the paper:

- Due to resource and time constraints, we performed most of experiments in the paper only once.  Therefore, the experimental results are somewhat randomized. And the parameter m of SOTA  is not well considered.
- Our training code is based on [**Face_Pytorch**](https://github.com/wujiyang/Face_Pytorch), and the data are taken from the maximum values during training. Even though no unfair comparisons occurred, we do not consider this approach to be very informative.

>  Now we have retrained some results according to the experimental setting in the paper for your reference. We get more conclusions than the paper.
>
>  - For fair comparison, all results are from the model with the highest average accuracy on all test sets.
>  - Related models can be found here [tod5](https://pan.baidu.com/s/1_KfiWWOw-FxMmi-1EaRYFQ?pwd=tod5).
>  - Test environment is  1 TitanX  torch 1.1.0 



**ratio: 0%：**

|       Method        |    LFW    |   AgeDB   |  CFP-FP   |   CALFW   |   CPLFW   |   SLLFW   |   Asian   | Caucasian |   India   |  African  |
| :-----------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|       ArcFace       |   99.28   |   93.5    |   94.89   |   93.17   |   89.13   |   97.55   |   85.57   |   92.88   |   89.9    |   86.45   |
|   MV-Arc-Softmax    |   99.23   |   93.83   |   94.61   |   93.17   |   89.35   |   97.88   |   85.8    |   93.18   |   90.23   |   86.35   |
|   CurricularFace    |   99.32   |   93.85   |   94.94   | **93.47** | **89.58** |   97.87   | **86.35** | **93.9**  | **90.45** | **87.57** |
|     BoundaryF1      |   99.28   |   94.13   |   94.66   |   92.8    |   89.27   |   97.88   |   85.6    |   92.95   |   89.72   |   86.27   |
| BoundaryFace（λ=π） | **99.37** | **94.32** | **94.99** |   93.15   |   89.27   | **98.12** |   85.97   |   93.75   |   89.73   |   87.12   |

The following conclusions can be drawn from the above data：

- When the proportion of closed-set noise in the training set is low, BoundaryF1 is comparable to ArcFace (baseline); in the CASIA-WebFace, BoundaryF1 ended up detecting only about 3800 closed-set noise samples.
- BoundaryFace is better than baseline and SOTA(MV-Arc-Softmax). BoundaryFace has a degree of advantage on the regular test set. SOTA(CurricularFace) has a relatively clear advantage in the RFW test set. In general, our approach is comparable to SOTA.

**ratio: 10%：**

|       Method        |    LFW    |   AgeDB   |  CFP-FP   |   CALFW   |  CPLFW   |   SLLFW   |   Asian   | Caucasian |  India   |  African  |
| :-----------------: | :-------: | :-------: | :-------: | :-------: | :------: | :-------: | :-------: | :-------: | :------: | :-------: |
|       ArcFace       |   99.1    |   93.78   |   94.49   |   93.15   |  89.03   |   97.8    |   85.33   |   93.22   |   89.9   |   86.75   |
|   MV-Arc-Softmax    | **99.35** |   94.18   |   94.27   | **93.42** |  89.28   |   97.73   |   86.03   | **93.6**  |  90.05   |   86.93   |
|   CurricularFace    |   99.17   |   93.63   |   93.63   |   93.07   |  88.63   |   97.65   |   85.05   |   92.68   |   89.7   |   86.37   |
|     BoundaryF1      |   99.3    |   93.68   | **94.79** |   93.23   | **89.5** |   97.78   |   85.85   |   93.4    |  89.78   |   86.88   |
| BoundaryFace（λ=π） |   99.33   | **94.18** |   94.64   |   93.3    |   89.2   | **97.85** | **86.18** |   93.55   | **90.4** | **87.53** |

**ratio: 20%：**

|       Method        |    LFW    |  AgeDB   |  CFP-FP  |   CALFW   |  CPLFW   |  SLLFW   |   Asian   | Caucasian |   India   |  African  |
| :-----------------: | :-------: | :------: | :------: | :-------: | :------: | :------: | :-------: | :-------: | :-------: | :-------: |
|       ArcFace       |   99.1    |    93    |   93.1   |   92.75   |  87.67   |  97.35   |   85.08   |   91.92   |   89.05   |   84.93   |
|   MV-Arc-Softmax    |   99.07   |  93.23   |  93.39   |   92.98   |  88.18   |  97.55   |   85.42   |   92.03   |   89.23   |   85.33   |
|   CurricularFace    |   98.97   |  91.63   |  92.11   |   92.03   |  87.35   |  96.32   |   84.35   |   90.7    |   87.87   |   83.37   |
|     BoundaryF1      |   99.22   |  93.88   | **94.2** |   93.48   | **88.6** | **97.9** |   85.95   |   93.02   |   89.5    | **87.05** |
| BoundaryFace（λ=π） | **99.25** | **93.9** |  93.99   | **93.48** |  88.15   |  97.78   | **86.37** | **93.47** | **89.63** |   87.02   |

The following conclusions can be drawn from the above data：

- The SOTA (CurricularFace) performance shows a very significant drop as the closed-set noise rate increases; when the dataset contains 10% closed-set noise, the drop in Baseline (ArcFace) performance is not yet significant, and SOTA (MV-Arc-Softmax) performance looks comparable to the performance without noise. When the dataset contains 20% closed-set noise, there is a significant degradation in the performance of all these methods compared to no noise.
- Unlike CurricularFace, MV-Arc-Softmax always outperforms ArcFace in general, even on noisy datasets.
- The performance of our method (BoundaryF1 and BoundaryFace) does not show a rapid degradation as the noise rate of the closed set increases. Even on a dataset containing 20% closed-set noise, our method can still maintain good performance.



### About the parameter m

 In this paper, we set the margin m = 0.5 for the training set containing closed-set noise ratio of 30% and open-set noise ratio of 10%, and we set the margin m = 0.3 for the other two mixing ratios.  Now, after our further experiments, we found that CurricularFace can get better results by setting m=0.5 on the other two mixing ratios. Besides, BoundaryF1 can obtain better results on C 20 O 20 by setting m=0.5.



**C: 20%  O: 20%**  (m=0.3)

|         Method         |    LFW    |   AgeDB   |  CFP-FP   |   CALFW   |   CPLFW   |   SLLFW   |   Asian   | Caucasian |   India   |  African  |
| :--------------------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|        ArcFace         |   98.75   |   90.13   |   89.56   |   90.83   |   83.9    |   95.07   |   81.5    |   88.45   |   85.7    |   79.92   |
|     MV-Arc-Softmax     |   98.75   |   89.13   |   89.2    |   90.65   |   84.45   |   95.15   |   81.23   |   88.8    |   85.48   |   79.47   |
| CurricularFace (m=0.5) |   98.32   |   89.63   |   89.94   |   90.87   |   84.47   |   94.58   |    81     |   88.22   |   85.28   |   78.83   |
|       BoundaryF1       |   98.97   |   92.3    |   89.5    |   91.82   |   83.73   |   96.75   |   84.2    |   90.95   | **88.55** |   83.87   |
|  BoundaryFace（λ=π）   | **99.13** | **92.63** | **92.89** | **92.43** | **87.03** | **97.03** | **84.55** | **91.63** |   88.27   | **84.77** |

Based on m = 0.5, we add the results of BoundaryF1:
|       Method       |  LFW  | AgeDB | CFP-FP | CALFW | CPLFW | SLLFW | Asian | Caucasian | India | African |
| :----------------: | :---: | :---: | :----: | :---: | :---: | :---: | :---: | :-------: | :---: | :-----: |
| BoundaryF1 (m=0.5) | 99.03 | 92.7  |   92   | 92.28 | 85.85 | 96.87 | 84.05 |   91.32   | 87.87 |  83.25  |



**C: 10%  O: 30%** (m=0.3)

|       Method        | LFW  | AgeDB | CFP-FP | CALFW | CPLFW | SLLFW | Asian | Caucasian | India | African |
| :-----------------: | :--: | :---: | :----: | :---: | :---: | :---: | :---: | :-------: | :---: | :-----: |
|       ArcFace       | 98.78 | 90.77 | 91.14 | 91.6 | 85.6 | 96.2 | 83.18 | 89.95 | 87.22 | 81.92 |
|   MV-Arc-Softmax    | 99.03 | 91.83 | 91.71 | 92.02 | 86.07 | 96.45 | 83.62 | 90.53 | 87.57 | 82.95 |
| CurricularFace (m=0.5) | 98.95 | 91.22 |   91.89   | 91.88 | 86.15 | 96.2 | 82.73 | 89.73 | 86.87 | 81.5 |
|     BoundaryF1      | 99.05 | 92.43 | 91.24 | 92.05 |   85.6    | 96.58 | 84.43 | **91.13** | 88.02 | 83.67 |
| BoundaryFace（λ=π） | **99.08** | **92.85** | **93.04** | **92.2** | **87.33** | **96.82** |**84.82**|91.07|**88.7**|**84.43**|

**Note:** Due to lack of rigor, we made a mistake on the experiment here. The results of CurricularFace in the paper for the C 10 O 30 case correspond to a parameter of m=0.5 instead of m=0.3. We did not examine the parameter settings carefully at that time, and this reimplement of the paper results led us to discover the problem.





**C: 30%  O: 10%** (m=0.5)

|       Method        | LFW  | AgeDB | CFP-FP | CALFW | CPLFW | SLLFW | Asian | Caucasian | India | African |
| :-----------------: | :--: | :---: | :----: | :---: | :---: | :---: | :---: | :-------: | :---: | :-----: |
|       ArcFace       | 98.53 | 89.5 | 88.11 | 90.85 | 82.47 | 94.77 | 81.17 | 87.82 | 85.68 | 79 |
|   MV-Arc-Softmax    | 98.53 | 89.58 | 87.86 | 90.88 | 82.37 | 95.02 | 80.62 | 88.1 | 85.18 | 79.12 |
|   CurricularFace    | 98.2 | 88.25 | 88.53 | 89.68 | 82.83 | 93.62 | 80 | 86.45 | 83.82 | 77.08 |
|     BoundaryF1      | **99.02** | 92.85 | **91.3** | 92.63 | **85.68** | 96.77 | **84.08** | 90.97 | **88.17** | 83.95 |
| BoundaryFace（λ=π） | 98.98 | **92.95** | 87.86 | **92.77** | 82.48 | **96.8** |83.55|**91.22**|87.9|**84.03**|




As can be seen from the tables above, our approach still has significant advantages over SOTA. Taken together, although BoundaryFace (λ = π) still outperforms BoundaryF1 on these noisy datasets, it is not as outstanding as in the paper (In the C30 O 10 case, BoundaryFace does not show a clear advantage over BoundaryF1). We believe that   the hyper-parameter λ may not be a good choice  in this case, and we will conduct more experiments for further exploration subsequently.



## Additional remarks

**Note on open-set noise in paper:**

Even though our method is for closed-set noise, experiments related to open-set noise are still performed.

In particular, we do not align open-set  noise samples (Only resize to 112 * 112 when used).

The main reason is as follows：

> In our initial experiments, we used open-set noise from distractors in the megaface dataset provided by insightface. However, during training, these aligned noisy samples may cause the training to crash. When we introduce only 20% of these open-set noise samples in the original WebFace, the accuracy of the model on the test set may occur as follows (Even if the m of ArcFace is reduced or softmax head is used):
>
> ![image](README.assets/image.svg) 
>
> This is incompatible with the situation caused by open-set noise in the real environment. The open set noise rate contained in MS1M is much higher than 20%, but the training process is still normal. The reasons for this problem, we think there may be two: 
>
> - The distribution of the simulated open-set noise samples may be very different from the distribution in the real environment.
> - The number of samples in WebFace is too small, if it is IMDB-Face or other Datasets may not appear this situation. 
>
> Such a bad situation does not occur if the unaligned samples are used as open-set noise sources, so this approach is used in the paper to simulate open-set noise in the real environment.





## Supplemental Results

Due to resource and time constraints, the method is not tested on the real dataset MS1M in our paper, which leads to a lower confidence level of the method. We understand if the paper was ultimately rejected for this reason, but the reviewers and the AC ultimately accepted our paper.

`In this subsection, we will supplement the experiments of our method on MS1M and provide all the data and files saved during the training process for your reference.`



### Visualization Results

#### Closed-set noise corrected by our method in WebFace

> The list of closed-set noise corrected during training is here[7dzp](https://pan.baidu.com/s/1VGio3-EsWeFY2-UDElLRfw).

**Example 1:**

![2022-08-24_214600](README.assets/2022-08-24_214600.png) 



**Example 2:**

![2022-08-24_214832](README.assets/2022-08-24_214832.png) 



**Example 3:**

![2022-08-24_215053](README.assets/2022-08-24_215053.png) 



**Example 4:**

![2022-08-24_215423](README.assets/2022-08-24_215423.png) 





#### Closed-set noise corrected by our method in MS1M

> MS1M is very noisy, and whether a sample is closed-set noise or not is entirely discerned by our method itself.
>
> - The closed-set noise label self-correction module is essentially based on model's generalization capability.

**Example 1:**

![2022-08-24_221314](README.assets/2022-08-24_221314.png) 



**Example 2:**

![2022-08-24_221540](README.assets/2022-08-24_221540.png) 



**Example 3:**

![2022-08-24_221759](README.assets/2022-08-24_221759.png) 





#### Closed-set noise corrected by our method in MS1MV2

> Even if MS1MV2 is considered as a clean dataset, our method still finds a small amount of closed-set noise.

**Example 1:**

![2022-08-24_222514](README.assets/2022-08-24_222514.png) 



**Example 2:**

![2022-08-24_223001](README.assets/2022-08-24_223001.png) 



**Example 3:**

![2022-08-24_223359](README.assets/2022-08-24_223359.png) 





### Test Results

> ***Note:***
>
> - All training logs and intermediate files on MS1M / MS1MV2 can be found here [pqff](https://pan.baidu.com/s/1bZnFjb_rVQZ68XDDioY_Kw).
> - All results reported in the tables are the performance at last epochs.
> - Test environment is 8 * RTX 3070 Pytorch 1.10.0 

**Training Set: MS1M**

IJB-C： 1:1 TAR @FAR=1e-4

|       Method       | MegaFace(R)@Rank1 | IJB-C     |    LFW    | AgeDB-30  |  CFP-FP   |   CALFW   |   CPLFW   |   SLLFW   |   Asian   | Caucasian |  Indian   |  African  |
| :----------------: | :---------------: | --------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|      ArcFace       |       96.45       | 91.65     |   99.7    |   97.55   | **96.43** |   95.92   |   91.53   |   99.3    |   95.13   |   98.4    |   96.75   |   96.02   |
|   MV-Arc-Softmax   |       96.86       | 91.47     | **99.72** |   97.72   |   96.4    |   95.85   | **92.02** |   99.28   |   95.15   | **98.42** |   96.73   |   96.33   |
|   CurricularFace   |       95.82       | 90.78     |   99.7    |   97.48   |   96.17   |   95.65   |   91.82   |   99.1    |   93.77   |   97.97   |   95.73   |   95.52   |
| BoundaryFace (λ=0) |     **97.57**     | **91.74** |   99.6    |   97.77   |   96.34   | **95.95** |   91.98   |   99.25   |   95.03   |   98.37   | **96.85** |   96.15   |
| BoundaryFace (λ=π) |       97.53       | 30.14     |   99.68   | **97.82** |   94.53   |   95.92   |   87.2    | **99.33** | **95.17** |   98.37   |   96.4    | **96.43** |

As can be seen from the table above, BoundaryFace(λ=π) performs very poorly on IJB-C even though it outperforms the baseline on MegaFace. We believe that the possible reasons are, on the one hand, that the hyper-parameter π is not applicable to BoundaryFace in real large-scale noisy datasets, and on the other hand, it may be caused by the excessive attention to open-set noise in the method itself. We will explore this problem in the future.



**Training Set: MS1MV2**

|       Method       | MegaFace(R)@Rank1 | IJB-C     |    LFW    | AgeDB-30  |  CFP-FP   |   CALFW   |   CPLFW   |   SLLFW   |   Asian   | Caucasian |  Indian   |  African  |
| :----------------: | :---------------: | --------- | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|      ArcFace       |       97.31       | **94.51** |   99.72   | **98.03** | **95.89** | **95.92** |   91.7    |   99.42   |    97     |   99.1    |   97.52   | **97.95** |
| BoundaryFace (λ=π) |     **97.41**     | 94.49     | **99.75** |   97.98   |   95.63   |   95.9    | **91.83** | **99.52** | **97.38** | **99.22** | **97.63** |   97.93   |



The following are the test results of the corresponding models on our private datasets. The datasets are collected in real environment.

|       Method       | S2V-s@Rank1 | S2V-v@Rank1 | Entry@Rank1 | HD@Rank1 | swjtu2D_SEN-s@Rank1 | swjtu2D_SEN-v@Rank1 |
| :----------------: | :---------: | :---------: | :---------: | :------: | :-----------------: | :-----------------: |
|      ArcFace       |    93.16    |  **72.43**  |    95.87    |  99.39   |        92.89        |        48.52        |
| BoundaryFace (λ=π) |  **95.44**  |    67.35    |  **96.7**   |  99.39   |      **93.91**      |      **51.9**       |

## Some conclusions

- Even though MS1M contains a lot of open-set noise, the performance of baseline on it does not degrade too much, and the training process does not crash.
- When dataset's noise rate is high, using BoundaryF1 (==BoundaryFace's special form (λ=0)==) can get good results.
- When dataset's noise rate is high, BoundaryFace(λ=π)'s performance on the CFP-FP and CPLFW  is much degraded  (It is also reflected in the paper, C 30 O 10) and the test curve fluctuates greatly during training. We believe this may be due to the distribution of noisy data or excessive noise rate causing problems in model learning.  We will explore the reasons for this in the future.
- When the dataset's noise rate is low (e.g. MS1MV2), using BoundaryFace can further improve performance.





# Todo

- Decrease the balance factor λ in formula to train BoundaryFace on MS1M / MS1MV2



# Acknowledgements

- BoundaryFace's motivation is partially  inspired by NPT-Loss  [Arxiv](https://arxiv.org/ftp/arxiv/papers/2103/2103.03503.pdf). And thanks the authors for their excellent work. 
- This code is largely based on [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo) and [**Face_Pytorch**](https://github.com/wujiyang/Face_Pytorch). We thank the authors a lot for their valuable efforts.

