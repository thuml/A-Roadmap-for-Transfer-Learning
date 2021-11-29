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

**Paper**
- Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
  [[ACL 2021]](https://aclanthology.org/2021.acl-long.568.pdf)
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [[ICLR 2021]](https://openreview.net/pdf?id=YicbFdNTTy) [[Tensorflow]](https://github.com/google-research/vision_transformer)
- Do Better ImageNet Models Transfer Better? [[CVPR 2019]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.pdf)
- Group Normalization [[ECCV 2018]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)
- Attention Is All You Need [[NIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Layer Normalization [[21 Jul 2016]] (https://arxiv.org/abs/1607.06450)
- Deep Residual Learning for Image Recognition [[CVPR 2016 Best]](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [[ICML 2015]](http://proceedings.mlr.press/v37/ioffe15.pdf)

**Library**
- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)

## Supervised Pre-Training

**Paper**
- Exploring the Limits of Large Scale Pre-training [[5 Oct 2021]](https://arxiv.org/abs/2110.02095)
- Do Adversarially Robust ImageNet Models Transfer Better? [[NIPS 2020]](https://proceedings.neurips.cc/paper/2020/file/24357dd085d2c4b1a88a7e0692e60294-Paper.pdf)
- Big Transfer (BiT): General Visual Representation Learning [[ECCV 2020]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500477.pdf) [[Code]](https://github.com/google-research/big_transfer)
- Billion-scale Semi-supervised Learning for Image Classification [[2 May 2019]](https://arxiv.org/abs/1905.00546) [[Pytorch]](https://github.com/leaderj1001/Billion-scale-semi-supervised-learning)
- ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness [[ICLR 2019]](https://openreview.net/pdf?id=Bygh9j09KX) [[Code]](https://github.com/rgeirhos/texture-vs-shape)
- Domain Adaptive Transfer Learning with Specialist Models [[16 Nov 2018]](https://arxiv.org/pdf/1811.07056.pdf)
- Exploring the Limits of Weakly Supervised Pretraining [[ECCV 2018]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dhruv_Mahajan_Exploring_the_Limits_ECCV_2018_paper.pdf) [[Pytorch]](https://github.com/facebookresearch/WSL-Images)

### Meta-Learning

**Survey**

**Paper**
- Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks [[ICML 2017]](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf) [[Tensorflow]](https://github.com/cbfinn/maml)  [[Pytorch]](https://github.com/dragen1860/MAML-Pytorch)

**Library**
- [learn2learn: A Library for {Meta-Learning} Research](https://github.com/learnables/learn2learn)


### Causal Learning
**Survey**

**Paper**
- Invariant Risk Minimization[[5 Jul 2019]] (https://arxiv.org/abs/1907.02893) [[Pytorch]](https://github.com/facebookresearch/InvariantRiskMinimization)

## Unsupervised Pre-Training
**Survey**

**Paper**

### Generative Learning
**Survey**

**Paper**
- Language Models are Few-Shot Learners [[NIPS 2020]](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) 
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [[NAACL 2019]](https://aclanthology.org/N19-1423.pdf) [[Tensorflow]](https://github.com/google-research/bert) 


### Contrastive Learning
**Survey**

**Paper**
- An Empirical Study of Training Self-Supervised Vision Transformers [[ICCV 2021 Oral]](https://arxiv.org/pdf/2104.02057.pdf) [[Pytorch]](https://github.com/facebookresearch/moco-v3)
- Exploring Simple Siamese Representation Learning [[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/facebookresearch/simsiam)
- A Simple Framework for Contrastive Learning of Visual Representations [[ICML 2020]](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) [[Tensorflow]](https://github.com/google-research/simclr) [[Pytorch]](https://github.com/sthalles/SimCLR)


## Task Adaptation

**Paper**
- Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
  [[ACL 2021]](https://aclanthology.org/2021.acl-long.568.pdf)
- How transferable are features in deep neural networks? [[NIPS 2014]](https://proceedings.neurips.cc/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf)
- DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition [[ICML 2014]](http://proceedings.mlr.press/v32/donahue14.pdf)
- OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks [[21 Dec 2013]](https://arxiv.org/abs/1312.6229)

### Catastrophic Forgetting
**Survey**

**Paper**
- Bi-tuning of pre-trained representations [[12 Nov 2020]](https://arxiv.org/abs/2011.06182) [[Translearners]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/task_adaptation/image_classification/bi_tuning.py)
- Stochastic Normalization [[NIPS 2020]](https://proceedings.neurips.cc/paper/2020/file/bc573864331a9e42e4511de6f678aa83-Paper.pdf) [[Pytorch]](https://github.com/thuml/StochNorm)
- Co-Tuning for Transfer Learning [[NIPS 2020]](https://papers.nips.cc/paper/2020/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf) [[Pytorch]](https://github.com/thuml/CoTuning)
- Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks [[ACL 2020]](https://aclanthology.org/2020.acl-main.740.pdf) [[Pytorch]](https://github.com/allenai/dont-stop-pretraining)
- SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization [[ACL 2020]](https://aclanthology.org/2020.acl-main.197.pdf) [[Pytorch]](https://github.com/namisan/mt-dnn)
- Theoretically Principled Trade-off between Robustness and Accuracy [[ICML 2019]](http://proceedings.mlr.press/v97/zhang19p/zhang19p.pdf) [[Pytorch]](https://github.com/yaodongyu/TRADES)
- DELTA: DEep Learning Transfer using Feature Map with Attention for Convolutional Networks [[ICLR 2019]](https://openreview.net/pdf?id=rkgbwsAcYm) [[Pytorch]](https://github.com/lixingjian/DELTA)
- An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models [[NAACL 2019]](https://aclanthology.org/N19-1213.pdf) [[Pytorch]](https://github.com/alexandra-chron/siatl)
- SpotTune: Transfer Learning through Adaptive Fine-tuning [[CVPR 2019]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_SpotTune_Transfer_Learning_Through_Adaptive_Fine-Tuning_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/gyhui14/spottune)
- Universal Language Model Fine-tuning for Text Classification [[ACL 2018]](https://aclanthology.org/P18-1031.pdf)
- Explicit Inductive Bias for Transfer Learning with Convolutional Networks [[ICML 2018]](http://proceedings.mlr.press/v80/li18a/li18a.pdf)
- Learning without Forgetting [[PAMI 2018]](https://arxiv.org/pdf/1606.09282.pdf)
- Overcoming catastrophic forgetting in neural networks [[PNAS 2017]](https://www.pnas.org/content/pnas/114/13/3521.full.pdf)

### Negative Transfer
**Survey**

**Paper**
- When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset [[ICAIL 2021]](https://arxiv.org/abs/2104.08671) [[Code]](https://github.com/reglab/casehold)
- Zoo-Tuning: Adaptive Transfer from A Zoo of Models [[ICML 2021]](http://proceedings.mlr.press/v139/shu21b/shu21b.pdf) [[Pytorch]](https://github.com/thuml/Zoo-Tuning)
- LogME: Practical Assessment of Pre-trained Models for Transfer Learning [[ICML 2021]](http://proceedings.mlr.press/v139/you21b/you21b.pdf) [[Pytorch]](https://github.com/thuml/LogME)
- LEEP: A New Measure to Evaluate Transferability of Learned Representations [[ICML 2020]](http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf)
- Rethinking ImageNet Pre-training [[ICCV 2019]](https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.pdf) 
- Characterizing and Avoiding Negative Transfer [[CVPR 2019]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.pdf)
- Catastrophic Forgetting Meets Negative Transfer [[NIPS 2019]](https://papers.nips.cc/paper/2019/file/c6bff625bdb0393992c9d4db0c6bbe45-Paper.pdf) [[Pytorch]](https://github.com/thuml/Batch-Spectral-Shrinkage)
- Learning What and Where to Transfer [[ICML 2019]](http://proceedings.mlr.press/v97/jang19b/jang19b.pdf) [[Pytorch]](https://github.com/alinlab/L2T-ww)
- Taskonomy: Disentangling Task Transfer Learning [[CVPR 2018 Best]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zamir_Taskonomy_Disentangling_Task_CVPR_2018_paper.pdf) [[Code]](https://github.com/StanfordVL/taskonomy)
- To Transfer or Not To Transfer [[NIPS 2005]](http://socrates.acadiau.ca/courses/comp/dsilver/ITWS05_Submitted_Papers/ITWS10-RosensteinM05_ITWS.pdf)

### Parameter Efficiency
**Survey**


**Paper**
- Parameter-Efficient Transfer Learning with Diff Pruning [[ACL 2021]](https://aclanthology.org/2021.acl-long.378.pdf) [[Pytorch]](https://github.com/dguo98/DiffPruning)
- BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning [[ICML 2019]](http://proceedings.mlr.press/v97/stickland19a/stickland19a.pdf) [[Pytorch]](https://github.com/AsaCooperStickland/Bert-n-Pals)
- Parameter-Efficient Transfer Learning for NLP [[ICML 2019]](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf) [[Tensorflow]](https://github.com/google-research/adapter-bert)
- Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks [[31 Dec 2019]](https://arxiv.org/abs/1912.13503) [[Pytorch]](https://github.com/jozhang97/side-tuning)
- Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights [[ECCV 2018]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.pdf) [[Pytorch]](https://github.com/arunmallya/piggyback)
- Learning multiple visual domains with residual adapters [[NIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/e7b24b112a44fdd9ee93bdf998c6ca0e-Paper.pdf)

### Data Efficiency
**Survey**
- Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing [[28 Jul 2021]](https://arxiv.org/abs/2107.13586)

**Paper**
- Finetuned Language Models Are Zero-Shot Learners [[3 Sep 2021]](https://arxiv.org/pdf/2109.01652v3.pdf) [[Tensorflow]](https://github.com/google-research/FLAN)
- Prefix-Tuning: Optimizing Continuous Prompts for Generation [[ACL 2021]](https://aclanthology.org/2021.acl-long.353.pdf) [[Pytorch]](https://github.com/XiangLi1999/PrefixTuning)
- Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference [[EACL 2020]](https://aclanthology.org/2021.eacl-main.20.pdf) [[Pytorch]](https://github.com/timoschick/pet)
- Language Models are Few-Shot Learners [[NIPS 2020]](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- A Closer Look at Few-shot Classification [[ICLR 2019]](https://openreview.net/pdf?id=HkxLXnAcFQ)
- Prototypical Networks for Few-shot Learning [[NIPS 2017]](https://papers.nips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf) [[Pytorch]](https://github.com/jakesnell/prototypical-networks)
- Matching Networks for One Shot Learning [[NIPS 2016]](https://papers.nips.cc/paper/2016/hash/90e1357833654983612fb05e3ec9148c-Abstract.html) [[Tensorflow]](https://github.com/AntreasAntoniou/MatchingNetworks)

**Library**
- [few-shot](https://github.com/oscarknagg/few-shot)

## Domain Adaptation

**Survey**
- A Survey on Transfer Learning [[KDE 2010]](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)

### Theory

**Paper**
- Bridging Theory and Algorithm for Domain Adaptation [[ICML 2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Pytorch]](https://github.com/thuml/MDD)
- Unsupervised Domain Adaptation Based on Source-guided Discrepancy [[AAAI 2019]](https://arxiv.org/abs/1809.03839)
- A theory of learning from different domains [[Machine Learning 2010]](https://www.alexkulesza.com/pubs/adapt_mlj10.pdf)
- Domain Adaptation: Learning Bounds and Algorithms [[COLT 2009]](https://www.learningtheory.org/colt2009/papers/003.pdf)
- Learning Bounds for Domain Adaptation [[NIPS 2007]](http://papers.nips.cc/paper/3212-learning-bounds-for-domain-adaptation)
- Analysis of Representations for Domain Adaptation [[NIPS 2006]](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation)

### Statistics Matching

**Paper**
- Representation Subspace Distance for Domain Adaptation Regression [[ICML 2021]](http://proceedings.mlr.press/v139/chen21u/chen21u.pdf) [[Pytorch]](https://github.com/thuml/Domain-Adaptation-Regression)
- Transferable Normalization: Towards Improving Transferability of Deep Neural Networks [[NIPS 2019]](https://papers.nips.cc/paper/2019/file/fd2c5e4680d9a01dba3aada5ece22270-Paper.pdf) [[Pytorch]](https://github.com/thuml/TransNorm)
- Contrastive Adaptation Network for Unsupervised Domain Adaptation [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/kgl-prml/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation)
- Adaptive Batch Normalization for practical domain adaptation [[Pattern Recognition 2018]](https://www.sciencedirect.com/science/article/pii/S003132031830092X)
- DeepJDOT: Deep Joint distribution optimal transport for unsupervised domain adaptation [[ECCV 2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf) [[Keras]](https://github.com/bbdamodaran/deepJDOT)
- Joint Distribution Optimal Transportation for Domain Adaptation [[NIPS 2017]](http://papers.nips.cc/paper/6963-joint-distribution-optimal-transportation-for-domain-adaptation.pdf) [[python]](https://github.com/rflamary/JDOT) [[Python Optimal Transport Library]](https://github.com/rflamary/POT)
- Central Moment Discrepancy for Unsupervised Domain Adaptation [[ICLR 2017]](https://openreview.net/pdf?id=SkB-_mcel), [[code]](https://github.com/wzell/cmd)
- Deep Transfer Learning with Joint Adaptation Networks [[ICML 2017]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-adaptation-networks-icml17.pdf) [[Translearners]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/jan.py)
- Deep CORAL: Correlation Alignment for Deep Domain Adaptation [[ECCV 2016]](https://arxiv.org/abs/1607.01719)
- Learning Transferable Features with Deep Adaptation Networks [[ICML 2015]](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) [[Translearners]](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/dan.py)
- Deep Domain Confusion: Maximizing for Domain Invariance [[Arxiv 2014]](https://arxiv.org/abs/1412.3474)
- Optimal kernel choice for large-scale two-sample tests [[NIPS 2012]](https://papers.nips.cc/paper/2012/file/dbe272bab69f8e13f14b405e038deb64-Paper.pdf)

### Domain Adversarial Learning

**Paper**
- Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation [[ICML 2019]](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) [[Pytorch]](https://github.com/thuml/Batch-Spectral-Penalization)
- Conditional Adversarial Domain Adaptation [[NIPS 2018]](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) [[Pytorch(official)]](https://github.com/thuml/CDAN)  [[Pytorch(third party)]](https://github.com/thuml/CDAN)
- Adversarial Discriminative Domain Adaptation [[CVPR2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)  [[Tensorflow(Official)]](https://github.com/erictzeng/adda) [[Pytorch]](https://github.com/corenel/pytorch-adda)
- Domain Separation Networks [[NIPS 2016]](http://papers.nips.cc/paper/6254-domain-separation-networks)
- Domain-Adversarial Training of Neural Networks [[JMLR 2016]](http://www.jmlr.org/papers/volume17/15-239/15-239.pdf)
- Simultaneous Deep Transfer Across Domains and Tasks [[ICCV 2015]](https://openaccess.thecvf.com/content_iccv_2015/papers/Tzeng_Simultaneous_Deep_Transfer_ICCV_2015_paper.pdf) 
- Unsupervised Domain Adaptation by Backpropagation [[ICML 2015]](http://proceedings.mlr.press/v37/ganin15.pdf) [[Caffe(Official)]](https://github.com/ddtm/caffe/tree/grl) [[Tensorflow]](https://github.com/shucunt/domain_adaptation) [[Pytorch]](https://github.com/fungtion/DANN)

**Application**
- Decoupled Adaptation for Cross-Domain Object Detection [[6 Oct 2021]](https://arxiv.org/abs/2110.02578)
- ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation [[CVPR 2019 Oral]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/valeoai/ADVENT)
- Strong-Weak Distribution Alignment for Adaptive Object Detection [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Saito_Strong-Weak_Distribution_Alignment_for_Adaptive_Object_Detection_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/VisionLearningGroup/DA_Detection)
- Domain Adaptive Faster R-CNN for Object Detection in the Wild [[CVPR 2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Domain_Adaptive_Faster_CVPR_2018_paper.pdf) [[Caffe2]](https://github.com/krumo/Detectron-DA-Faster-RCNN) [[Caffe]](https://github.com/yuhuayc/da-faster-rcnn) [[Pytorch(under developing)]]()
- Learning to Adapt Structured Output Space for Semantic Segmentation [[CVPR 2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tsai_Learning_to_Adapt_CVPR_2018_paper.pdf) [[Pytorch]](https://github.com/wasidennis/AdaptSegNet)
- FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation [[8 Dec 2016]](https://arxiv.org/abs/1612.02649)


### Hypothesis Adversarial Learning
**Paper**
- Regressive Domain Adaptation for Unsupervised Keypoint Detection [[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Jiang_Regressive_Domain_Adaptation_for_Unsupervised_Keypoint_Detection_CVPR_2021_paper.pdf) [[Pytorch]](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/keypoint_detection)
- Bridging Theory and Algorithm for Domain Adaptation [[ICML 2019]](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) [[Pytorch]](https://github.com/thuml/MDD)
- Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lee_Sliced_Wasserstein_Discrepancy_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf)
- Maximum Classifier Discrepancy for Unsupervised Domain Adaptation [[CVPR 2018]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) [[Pytorch(Official)]](https://github.com/mil-tokyo/MCD_DA)

### Domain Translation

**Paper**
- Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection [[CVPR 2019]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Diversify_and_Match_A_Domain_Adaptive_Representation_Learning_Paradigm_for_CVPR_2019_paper.pdf) [[Pytorch]](https://github.com/TKKim93/DivMatch)
- CyCADA: Cycle-Consistent Adversarial Domain Adaptation [[ICML 2018]](http://proceedings.mlr.press/v80/hoffman18a.html) [[Pytorch(official)]](https://github.com/jhoffman/cycada_release)
- Using simulation and domain adaptation to improve efficiency of deep robotic grasping [[ICRA 2018]](https://arxiv.org/abs/1709.07857)
- Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[CVPR 2018]](https://arxiv.org/abs/1704.01705) [[Pytorch(Official)]](https://github.com/yogeshbalaji/Generate_To_Adapt)
- Person Transfer GAN to Bridge Domain Gap for Person Re-Identification [[CVPR 2018]](https://arxiv.org/abs/1711.08565v2)
- Unsupervised Machine Translation Using Monolingual Corpora Only [[ICLR 2017]](https://openreview.net/pdf?id=rkYTTf-AZ) 
- Unsupervised Cross-Domain Image Generation [[ICLR 2017]](https://openreview.net/forum?id=Sk2Im59ex) [[TensorFlow]](https://github.com/yunjey/domain-transfer-network)
- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks [[ICCV 2017]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf) [[Pytorch(Official)]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks [[CVPR 2017]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bousmalis_Unsupervised_Pixel-Level_Domain_CVPR_2017_paper.pdf) [[Tensorflow(Official)]](https://github.com/tensorflow/models/tree/master/research/domain_adaptation) [[Pytorch]](https://github.com/vaibhavnaagar/pixelDA_GAN)
- Learning from Simulated and Unsupervised Images through Adversarial Training [[CVPR 2017 Oral]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf) [[Tensorflow]](https://github.com/carpedm20/simulated-unsupervised-tensorflow)
- Coupled Generative Adversarial Networks [[NIPS 2016]](http://papers.nips.cc/paper/6544-coupled-generative-adversarial-networks) [[Pytorch(Official)]](https://github.com/mingyuliutw/cogan)
- Generative Adversarial Nets [[NIPS 2014]](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)

### Semi-Supervised Learning

**Paper**
- Cycle Self-Training for Domain Adaptation [[NIPS 2021]](https://papers.nips.cc/paper/2021/file/c1fea270c48e8079d8ddf7d06d26ab52-Paper.pdf) [[Pytorch]](https://github.com/Liuhong99/CST)
- Adapting ImageNet-scale models to complex distribution shifts with self-learning [[27 Apr 2021]](https://arxiv.org/abs/2104.12928)
- Minimum Class Confusion for Versatile Domain Adaptation [[ECCV 2020]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660460.pdf)
- Semi-supervised Domain Adaptation via Minimax Entropy [[ICCV 2019]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Saito_Semi-Supervised_Domain_Adaptation_via_Minimax_Entropy_ICCV_2019_paper.pdf) [[Pytorch]](https://github.com/VisionLearningGroup/SSDA_MME)
- Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification [[ICLR 2020]](https://openreview.net/forum?id=rJlnOhVYPS) [[Pytorch]](https://github.com/yxgeee/MMT)
- Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels [[NIPS 2018]](https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf)
- Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training [[ECCV 2018]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf) [[Pytorch]](https://github.com/yzou2/CBST)
- A DIRT-T Approach to Unsupervised Domain Adaptation [[ICLR 2018]](https://openreview.net/forum?id=H1q-TM-AW) [[Tensorflow(Official)]](https://github.com/RuiShu/dirt-t)
- Self-Ensembling for Visual Domain Adaptation [[ICLR 2018]](https://openreview.net/forum?id=rkpoTaxA-)
- Asymmetric Tri-training for Unsupervised Domain Adaptation [[ICML 2017]](http://proceedings.mlr.press/v70/saito17a.html) [[TensorFlow]](https://github.com/ksaito-ut/atda)

## Evaluation

### Cross-Task Evaluation

**Dataset**

**Library**

### Cross-Domain Evaluation

**Dataset**

- XNLI: Evaluating Cross-lingual Sentence Representations [[EMNLP 2018]](https://aclanthology.org/D18-1269.pdf) [[Download]](https://github.com/facebookresearch/XNLI)


**Library**






