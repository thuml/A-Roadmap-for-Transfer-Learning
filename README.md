# Awesome-Transferability-in-Deep-Learning

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repo is a collection of AWESOME things about transferablity in deep learning, including papers, code, etc. Feel free to star and fork.

# Contents

- [Contents](#contents)
- [Introduction](#introduction)
- [Pre-Training Models](#pre-training-models)
- [Supervised Pre-Training](#supervised-pre-training)
    - [Meta-Learning](#meta-learning)
    - [Causal Learning](#causal-learning)
- [Unsupervised Pre-Training](#unsupervised-pre-training)
    - [Generative Learning](#generative-learning)
    - [Contrastive Learning](#contrastive-learning)
- [Task Adaptation](#task-adaptation)
    - [Catastrophic Forgetting](#catastrophic-forgetting)
    - [Negative Transfer](#negative-transfer)
    - [Parameter Efficiency](#parameter-efficiency)
    - [Data Efficiency](#data-efficiency)    
- [Domain Adaptation](#domain-adaptation)
    - [Theory](#theory)
    - [Statistics Matching](#statistics-matching)
    - [Domain Adversarial Learning](#domain-adversarial-learning)
    - [Hypothesis Adversarial Learning](#hypothesis-adversarial-learning)
    - [Domain Translation](#domain-translation)
    - [Semi-Supervised Learning](#semi-supervised-learning)
- [Evaluation](#evaluation)
    - [Cross-Task Evaluation](#cross-task-evaluation)
    - [Cross-Domain Evaluation](#cross-domain-evaluation)


[comment]: <> (Theory, Survey, Benchmarks, Library, Dataset, Lectures and Tutorials)

## Introduction

## Pre-Training Models

**Survey**

**Paper**

**Library**

## Supervised Pre-Training

**Survey**

**Paper**

### Meta-Learning

**Survey**

**Paper**

### Causal Learning
**Survey**

**Paper**

## Unsupervised Pre-Training
**Survey**

**Paper**

### Generative Learning
**Survey**

**Paper**

### Contrastive Learning
**Survey**

**Paper**

## Task Adaptation

### Catastrophic Forgetting
**Survey**

**Paper**

### Negative Transfer
**Survey**

**Paper**

### Parameter Efficiency
**Survey**

**Paper**

### Data Efficiency
**Survey**

**Paper**

## Domain Adaptation

### Theory
**Survey**

**Paper**
- Bridging Theory and Algorithm for Domain Adaptation [[ICML2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Pytorch]](https://github.com/thuml/MDD)
- On Learning Invariant Representation for Domain Adaptation [[ICML2019]](https://arxiv.org/abs/1901.09453v1) [[code]](https://github.com/KeiraZhao/On-Learning-Invariant-Representations-for-Domain-Adaptation)
- Unsupervised Domain Adaptation Based on Source-guided Discrepancy [[AAAI2019]](https://arxiv.org/abs/1809.03839)
- Learning Bounds for Domain Adaptation [[NIPS2007]](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
- Analysis of Representations for Domain Adaptation [[NIPS2006]](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation)

### Statistics Matching
**Survey**

**Paper**
- Adaptive Batch Normalization for practical domain adaptation [[Pattern Recognition(2018)]](https://www.sciencedirect.com/science/article/pii/S003132031830092X)
- DeepJDOT: Deep Joint distribution optimal transport for unsupervised domain adaptation [[ECCV2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf) [[Keras]](https://github.com/bbdamodaran/deepJDOT)
- Joint Distribution Optimal Transportation for Domain Adaptation [[NIPS2017]](http://papers.nips.cc/paper/6963-joint-distribution-optimal-transportation-for-domain-adaptation.pdf) [[python]](https://github.com/rflamary/JDOT) [[Python Optimal Transport Library]](https://github.com/rflamary/POT)
- Central Moment Discrepancy for Unsupervised Domain Adaptation [[ICLR2017]](https://openreview.net/pdf?id=SkB-_mcel), [[InfSc2019]](https://arxiv.org/pdf/1711.06114.pdf), [[code]](https://github.com/wzell/cmd)
- Deep Transfer Learning with Joint Adaptation Networks [[ICML2017]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf) [[code]](https://github.com/thuml/Xlearn)
- Deep CORAL: Correlation Alignment for Deep Domain Adaptation [[ECCV2016]](https://arxiv.org/abs/1607.01719)
- Learning Transferable Features with Deep Adaptation Networks [[ICML2015]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) [[code]](https://github.com/thuml/DAN)
- Deep Domain Confusion: Maximizing for Domain Invariance [[Arxiv 2014]](https://arxiv.org/abs/1412.3474)


### Domain Adversarial Learning
**Survey**

**Paper**
- Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation [[ICML2019]](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) [[Pytorch]](https://github.com/thuml/Batch-Spectral-Penalization)
- Conditional Adversarial Domain Adaptation [[NIPS2018]](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) [[Pytorch(official)]](https://github.com/thuml/CDAN)  [[Pytorch(third party)]](https://github.com/thuml/CDAN)
- A DIRT-T Approach to Unsupervised Domain Adaptation [[ICLR2018 Poster]](https://openreview.net/forum?id=H1q-TM-AW) [[Tensorflow(Official)]](https://github.com/RuiShu/dirt-t)
- Adversarial Discriminative Domain Adaptation [[CVPR2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)  [[Tensorflow(Official)]](https://github.com/erictzeng/adda) [[Pytorch]](https://github.com/corenel/pytorch-adda)
- Domain Separation Networks [[NIPS2016]](http://papers.nips.cc/paper/6254-domain-separation-networks)
- Domain-Adversarial Training of Neural Networks [[JMLR2016]](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)
- Unsupervised Domain Adaptation by Backpropagation [[ICML2015]](http://proceedings.mlr.press/v37/ganin15.pdf) [[Caffe(Official)]](https://github.com/ddtm/caffe/tree/grl) [[Tensorflow]](https://github.com/shucunt/domain_adaptation) [[Pytorch]](https://github.com/fungtion/DANN)

**Application**
- ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation [[CVPR2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/valeoai/ADVENT)
- Strong-Weak Distribution Alignment for Adaptive Object Detection [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Saito_Strong-Weak_Distribution_Alignment_for_Adaptive_Object_Detection_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/VisionLearningGroup/DA_Detection)
- Domain Adaptive Faster R-CNN for Object Detection in the Wild [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf) [[Caffe2]](https://github.com/krumo/Detectron-DA-Faster-RCNN) [[Caffe]](https://github.com/yuhuayc/da-faster-rcnn) [[Pytorch(under developing)]]()
- Learning to Adapt Structured Output Space for Semantic Segmentation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tsai_Learning_to_Adapt_CVPR_2018_paper.pdf) [[Pytorch]](https://github.com/wasidennis/AdaptSegNet)


### Hypothesis Adversarial Learning
**Survey**

**Paper**
- Bridging Theory and Algorithm for Domain Adaptation [[ICML2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Pytorch]](https://github.com/thuml/MDD)
- Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation [[CVPR2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Sliced_Wasserstein_Discrepancy_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
- Maximum Classifier Discrepancy for Unsupervised Domain Adaptation [[CVPR2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/mil-tokyo/MCD_DA)

### Domain Translation
**Survey**

**Paper**

- CyCADA: Cycle-Consistent Adversarial Domain Adaptation [[ICML2018]](http://proceedings.mlr.press/v80/hoffman18a.html) [[Pytorch(official)]](https://github.com/jhoffman/cycada_release)
- Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[CVPR2018]](https://arxiv.org/abs/1704.01705) [[Pytorch(Official)]](https://github.com/yogeshbalaji/Generate_To_Adapt)
- Unsupervised Image-to-Image Translation Networks [[NIPS2017]](http://papers.nips.cc/paper/6672-unsupervised-image-to-image-translation-networks) [[Pytorch(Official)]](https://github.com/mingyuliutw/unit)
- Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks [[CVPR2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.pdf) [[Tensorflow(Official)]](https://github.com/tensorflow/models/tree/master/research/domain_adaptation) [[Pytorch]](https://github.com/vaibhavnaagar/pixelDA_GAN)
- Unsupervised Cross-Domain Image Generation [[ICLR2017 Poster]](https://openreview.net/forum?id=Sk2Im59ex) [[TensorFlow]](https://github.com/yunjey/domain-transfer-network)
- Coupled Generative Adversarial Networks [[NIPS2016]](http://papers.nips.cc/paper/6544-coupled-generative-adversarial-networks) [[Pytorch(Official)]](https://github.com/mingyuliutw/cogan)


### Semi-Supervised Learning
**Survey**

**Paper**

- Minimum Class Confusion for Versatile Domain Adaptation [[ECCV2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf)
- Semi-supervised Domain Adaptation via Minimax Entropy [[ICCV2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/VisionLearningGroup/SSDA_MME)
- Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification [[ICLR2020]](https://openreview.net/forum?id=rJlnOhVYPS) [[Pytorch]](https://github.com/yxgeee/MMT)
- Self-Ensembling for Visual Domain Adaptation [[ICLR2018]](https://openreview.net/forum?id=rkpoTaxA-)
- Asymmetric Tri-training for Unsupervised Domain Adaptation [[ICML2017]](http://proceedings.mlr.press/v70/saito17a.html) [[TensorFlow]](https://github.com/ksaito-ut/atda)

## Evaluation

### Cross-Task Evaluation

**Dataset**

**Library**

### Cross-Domain Evaluation

**Dataset**

**Library**






