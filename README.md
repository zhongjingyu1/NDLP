# NDLP
#### Noise Correction and Distribution Fine-Tuning for Long-Tailed Partial Multi-Label Learning

![long](https://img.shields.io/badge/State%20of%20the%20Art-Long--Tailed%20Partial%20Multi--Label%20Learning%20on%20VOC--MLC-blue)  
![long](https://img.shields.io/badge/State%20of%20the%20Art-Long--Tailed%20Partial%20Multi--Label%20Learning%20on%20COCO--MLC-blue)  

## :video_camera: Problem description of LT-PML
In this paper, we present a new challenge for MLC on data setup called long-tailed partial multi-label learning (LT-PML), with both PML setup and LT distribution problems. As shown in the overview of LT-PML in Figure 1, LT-PML has the following two challenges: 1) `Mutual hindrance of noisy multi-label and long-tailed distributions`. First, the noisy candidate label sets prevent obtaining the class-wise label frequency priors, yet which is critical for existing LT methods. In addition, noisy labels are biased towards sparser tail classes due to ambiguity. This is because the classes that the labeler cannot accurately judge empirically after being misled by ambiguity often belong to infrequently labeled classes. The LT distribution skews the distribution, causing the model learning to be heavily biased towards the head class, which leads to the difficulty of label disambiguation (Figure 1(b)). 2) `Label co-occurrence`. Label co-occurrence is common in natural images. As shown in Figure 1(c), the rare label pomegranate appears in the same sample as the common label computer, which leads to the fact that resampling such an image does not necessarily improve the unbalanced class distribution and indirectly impairs the head class performance in training.
<img src='./assets/intr1.png' width=800>

## :scroll: Abstract 
Long-tailed multi-label classification (LT-MLC) assumes that all samples are noise-free. The presence of label noise makes the prior class distribution unreliable. To address this problem, we introduce a new task, long-tailed partial multi-label learning (LT-PML), to consider noisy learning environments. LT-PML aims to generalize a classifier from long-tailed training samples, where each sample is associated with a set of candidate labels and only some of the labels are accurate. Not surprisingly, we find that the performance of most MLC and LT-MLC methods degrades significantly in the face of this task. Therefore, we propose a Noise correction and Distribution fine-tuning framework for LT-PML (NDLP). First, we estimate the confidence level of each label as the ground truth to mitigate the noisy label interference, and further use it to match the class distribution. In addition, we propose a noise correction method that uses prediction probabilities to re-weight classes to mitigate the negative contribution of the noisy labels to the positive samples. Finally, we develop a distributional fine-tuning to correct the estimation error due to label co-occurrence. The results indicate that NDLP significantly outperforms existing methods on LT-PML datasets with noise.

<img src='./assets/FLOW.png' width=1200>

## :closed_book: Requirements 
* [Pytorch](https://pytorch.org/)
* [Sklearn](https://scikit-learn.org/stable/)
## :clipboard: Dataset 
To evaluate/train Long-Tailed Partial Multi-Label Learning, first, the [VOC2012/2007](http://host.robots.ox.ac.uk/pascal/VOC/) and [MSCOCO](https://cocodataset.org/#download) datasets need to be downloaded. The image paths, labels, and captions for the VOC-MLT and COCO-MLT datasets can be found [here]().

* [COCO-MLT](https://github.com/wutong16/DistributionBalancedLoss/tree/master/appendix/coco)
* [VOC-MLT](https://github.com/wutong16/DistributionBalancedLoss/tree/master/appendix/VOCdevkit)
```
Shell
├── appendix
    ├── coco
        ├── coco_lt_train.txt
        ├── coco_lt_test.txt
        ├── coco_labels.txt
        ├── longtail2017
            ├── class_freq.pkl
            ├── class_split.pkl
    ├── VOCdevkit
        ├── voc_lt_train.txt
        ├── voc_lt_test.txt
        ├── voc_labels.txt
        ├── longtail2012
            ├── class_freq.pkl
            ├── class_split.pkl
├── data
    ├── coco
        ├── train2017
            ├── 0000001.jpg
            ...
        ├── val2017
            ├── 0000002.jpg
            ...
    ├── voc
        ├── VOCdevkit
            ├── VOC2007
                ├── Annotations
                ├── ImageSets
                ├── JPEGImages
                    ├── 0000001.jpg
                    ...
                ├── SegementationClass
                ├── SegementationObject
            ├── VOC2012
                ├── Annotations
                ├── ImageSets
                ├── JPEGImages
                    ├── 0000002.jpg
                    ...
                ├── SegementationClass
                ├── SegementationObject
```
## :page_with_curl: Usage
### VOC-MLT
```bash
python lmpt/train.py configs/voc/LT_resnet50_pfc_DB.py \
--dataset 'voc-lt' \
--seed '0' \
--batch_size 32 \
--epochs 10 \
--gamma 3
--partial_rate 0.3/0.5 \
--eta 0.9 \
--alpha_range 0.4,0.8 \
```
### COCO-MLT
```bash
python lmpt/train.py configs/coco/LT_resnet50_pfc_DB.py \ 
--dataset 'voc-lt' \
--seed '0' \
--batch_size 32 \
--epochs 10 \
--gamma 2
--partial_rate 0.05/0.1 \
--eta 0.9 \
--alpha_range 0.4,0.8 \
```
## :wrench: Pre-trained models
#### COCO-MLT
|   Backbone  | $$\rho$$  |    Total   |    Head   |  Medium  |   Tail  |      Download      |
| :---------: |:---------:| :------------: | :-----------: | :---------: | :---------: | :----------------: |
|  ResNet-50  |    0.05   |    40.06   |    41.87  |  40.57   | 37.80   | [model](https://drive.google.com/drive/folders/1ju2zTv6pOuso8wBixN4RqLtszU8d8WDk?hl=zh-cn)   |
|  ResNet-50  |    0.1    |    35.21   |    40.65  |  32.83   | 33.55   | [model](https://drive.google.com/drive/folders/1ju2zTv6pOuso8wBixN4RqLtszU8d8WDk?hl=zh-cn)   |

#### VOC-MLT
|   Backbone  | $$\rho$$  |    Total   |    Head   |  Medium  |   Tail  |      Download      |
| :---------: |:---------:| :------------: | :-----------: | :---------: | :---------: | :----------------: |
|  ResNet-50  |    0.3    |    59.51   |    59.54  |  70.33   | 51.38   | [model](https://drive.google.com/drive/folders/1ju2zTv6pOuso8wBixN4RqLtszU8d8WDk?hl=zh-cn)   |
|  ResNet-50  |    0.5    |    40.76   |    43.09  |  52.59   | 30.13   | [model](https://drive.google.com/drive/folders/1ju2zTv6pOuso8wBixN4RqLtszU8d8WDk?hl=zh-cn)   |
## :heartpulse: Acknowledgements
We use code from [DBL](https://github.com/wutong16/DistributionBalancedLoss) and [LMPT](https://github.com/richard-peng-xia/LMPT). We thank the authors for releasing their code.
## :mailbox_closed: Contact
If you have any questions, please create an issue on this repository or contact at [23171214508@stu.xidian.edu.cn](mailto:23171214508@stu.xidian.edu.cn).
<!-- ## :pencil2: Citing
If you find this code useful, please consider to cite our work.
```
``` -->



**To reviewer p11R:**

**Q1:** Insightful comment! Table 1 correlation measure shows that CoNeS is better than the initial correlation. Validity is confirmed by the performance drop by the initial matrix (original Table 4). Our original statement is inaccurate and will be corrected, including MLIC work discussion. Original Tables 1 and 2 clearly show that direct transfer ASL is invalid, indicating the need for an NSL designed for PMLIC.

| Method             | VOC2007        |                | MS-COCO        |                |
|--------------------|----------------|----------------|----------------|----------------|
|                    | Cos            | Ssim           | Cos            | Ssim           |
|--------------------|----------------|----------------|----------------|----------------|
| Initial correlation| 0.9086         | 0.5852         | 0.7669         | 0.2099         |
| **CoNeS**          | **0.9801**     | **0.7801**     | **0.9497**     | **0.4413**     |

*Table 1: Correlation measure with truth label co-occurrence (Cos = cosine similarity, Ssim = structural similarity)*

---

**Q2:** Insightful comment! We carefully explored your suggestion with Table 2: Outlier-resistant DBSCAN substitution yields no benefit. We interpret that most misattributed features are pre-filtered by CAM, mitigating k-means outlier sensitivity. We also evaluated initial value-optimized k-means++. Surprisingly, it offered no advantage. We think that the slight randomness is analogous to implicit regularization precisely circumvents the fixation on learning biased patterns.

| Method       | mAP   | CF1   | OF1   |
|--------------|-------|-------|-------|
| w. k-means++ | 91.80 | 86.41 | 89.44 |
| w. DBSCAN    | 91.76 | 86.55 | 89.38 |
| **CoNeS**    | 92.10 | 86.26 | 89.16 |

*Table 2: Ablation study for k-means*

---

**Q3:** Based on your comments, "dispersed" misleadingly emphasizes spatial spread over activation strength, so we will replace it with "weakly activated". "Corrupted pixels" incorrectly implies physical damage; we will use "misattributed pixel regions" instead to denote regions the model activates incorrectly.

**Q4:** Insightful comments! We acknowledge that expanding the connections to broader CAM-based disambiguation and negative suppression would strengthen the innovation.

**Q5:** We appreciate this concern. Clarifying, forcing uniform settings would be unreasonable. Take TPML: it requires 224 resolution to maintain computational efficiency for its linear models. Further, we strictly follow each baseline's original setup. Changing these could hurt performance by misleading true capabilities. This practice is standard in MLIC research ([65,74,81]), preserving evaluation integrity.


