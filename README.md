## [Deep Learning for Video Object Segmentation: A Review]() (TODO: paper link)

This repo is built to facilitate access to the related VOS datasets and papers (with code links if applicable). 

中文版博客[链接]() (TODO: blog link)

If you find our review and repository useful for your research, please consider citing our paper:

```bibtex
TODO: article bibtex
```

## Content

- [Section 3. Datasets](#section-3-datasets)
- [Section 4. Methods](#section-4-methods)
  - [4.1. Online Fine-tuning-based VOS](#41-online-fine-tuning-based-vos)
  - [4.2. Feature Matching-based VOS](#42-feature-matching-based-vos)
  - [4.3. Graph Optimisation-based VOS](#43-graph-optimisation-based-vos)
  - [4.4. Optical Flow-based VOS](#44-optical-flow-based-vos)
  - [4.5. Mask Propagation-based VOS](#45-mask-propagation-based-vos)
  - [4.6. Long-term Propagation-based VOS](#46-long-term-propagation-based-vos)

## Section 3. Datasets

#### Earlier datasets for VOS evaluation
|Years|Datasets with links|Paper links|
|:-:|---|:-:|
|2007|[Hopkins 155](http://www.vision.jhu.edu/data/hopkins155/)|[link](https://www.cis.jhu.edu/~rvidal/publications/cvpr07-benchmark.pdf)|
|2010|[BMS-26](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html)|[link](https://link.springer.com/content/pdf/10.1007/978-3-642-15555-0_21.pdf)|
|2013|[FBMS-59](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html)|[link](https://ieeexplore.ieee.org/document/6682905)|
|2012|[SegTrack v1](https://cpl.cc.gatech.edu/projects/SegTrack/)|[link](http://www.bmva.org/bmvc/2010/conference/paper56/paper56.pdf)|
|2013|[SegTrack v2](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html)|[link](https://ieeexplore.ieee.org/document/6751383)|
|2012|[YouTube-Objects](https://vision.cs.utexas.edu/projects/videoseg/)|[link](https://www.cs.utexas.edu/~grauman/papers/suyog-eccv2014.pdf)|
|2015|[JumpCut](https://www.dropbox.com/s/v0v3pkrhz1vizyt/VideoSeg_dataset.rar?dl=0)|[link](https://dl.acm.org/doi/10.1145/2816795.2818105)|

#### :fire: Popular datasets in deep learning era

|Years|Datasets with links|Paper links|Remarks|
|:-:|---|:-:|---|
|2016|[DAVIS-2016](https://davischallenge.org/davis2016/code.html)|[link](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)|(Single object) Applicable to train and evaluate both Unsupervised and Semi-supervised VOS methods|
|2017|[DAVIS-2017](https://davischallenge.org/davis2017/code.html)|[link](https://arxiv.org/pdf/1704.00675.pdf)|(Multiple objects) Applicable to train and evaluate Semi-supervised VOS methods (Two popular subsets for evaluation: validation and test-dev. The latter is more challenging.)|
|2019|[DAVIS-2017-U](https://davischallenge.org/davis2017/code.html)|[link](https://arxiv.org/pdf/1704.00675.pdf)|(Multiple objects) Applicable to train and evaluate Unsupervised VOS methods|
|2018|[YouTube-VOS-2018](https://competitions.codalab.org/competitions/19544#participate-get-data)|[link](https://arxiv.org/pdf/1809.03327.pdf)|(Multiple objects) Applicable to train and evaluate Semi-supervised VOS methods (Registration is required when downloading YouTube-VOS/VIS data)|
|2019|[YouTube-VOS-2019](https://competitions.codalab.org/competitions/20127#participate-get-data)|[link](https://arxiv.org/pdf/1809.03327.pdf)|(Multiple objects) Applicable to train and evaluate Semi-supervised VOS methods|
|2019|[YouTube-VIS-2019](https://competitions.codalab.org/competitions/20128#participate-get_data)|[link](https://arxiv.org/pdf/1905.04804.pdf)|(Multiple objects) Applicable to train and evaluate Unsupervised VOS methods|
|2021|[YouTube-VIS-2021](https://competitions.codalab.org/competitions/28988#participate-get_data)|[link](https://arxiv.org/pdf/1905.04804.pdf)|(Multiple objects) Applicable to train and evaluate Unsupervised VOS methods|




## Section 4. Methods
TODO: tables showing the pdf and code links of the reviewed papers

### 4.1. Online Fine-tuning-based VOS

|Paper titles with links (abbreviates in our review)|Venues|Years|Codes|
|---|:-:|:-:|:-:|
|[One-Shot Video Object Segmentation](https://openaccess.thecvf.com/content_cvpr_2017/papers/Caelles_One-Shot_Video_Object_CVPR_2017_paper.pdf) (OSVOS)|CVPR|2017|[PyTorch](https://github.com/kmaninis/OSVOS-PyTorch)|
|[Online Adaptation of Convolutional Neural Networks for Video Object Segmentation](https://arxiv.org/pdf/1706.09364.pdf) (OnAVOS)|BMVC|2017|[Tenserflow](https://www.vision.rwth-aachen.de/page/OnAVOS)|
|[Video Object Segmentation without Temporal Information](https://ieeexplore.ieee.org/document/8362936) (OSVOS-S)|TPAMI|2018|[Website](https://cvlsegmentation.github.io/osvos-s/)|
|[Video Object Segmentation by Learning Location-Sensitive Embeddings](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Hai_Ci_Video_Object_Segmentation_ECCV_2018_paper.pdf) (LSE-VOS)|ECCV|2018||
|[Lucid Data Dreaming for Video Object Segmentation](https://link.springer.com/article/10.1007/s11263-019-01164-6)|IJCV|2019|[MATLAB](https://github.com/ankhoreva/LucidDataDreaming)|
|[BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames](https://openaccess.thecvf.com/content_CVPR_2019/papers/Griffin_BubbleNets_Learning_to_Select_the_Guidance_Frame_in_Video_Object_CVPR_2019_paper.pdf)|CVPR|2019|[Tensorflow](https://github.com/griffbr/BubbleNets)|

#### Variants

|Paper titles with links (abbreviates in our review)|Venues|Years|Codes|
|---|:-:|:-:|:-:|
|[Efficient Video Object Segmentation via Network Modulation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Efficient_Video_Object_CVPR_2018_paper.pdf) (OSMN)|CVPR|2018|[Tensorflow](https://github.com/linjieyangsc/video_seg)|
|[A Generative Appearance Model for End-to-end Video Object Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Johnander_A_Generative_Appearance_Model_for_End-To-End_Video_Object_Segmentation_CVPR_2019_paper.pdf) (AGAME)|CVPR|2019|[PyTorch](https://github.com/joakimjohnander/agame-vos)|
|[Learning Fast and Robust Target Models for Video Object Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Robinson_Learning_Fast_and_Robust_Target_Models_for_Video_Object_Segmentation_CVPR_2020_paper.pdf) (FRTM)|CVPR|2020|[PyTorch](https://github.com/andr345/frtm-vos)|
|[Learning What to Learn for Video Object Segmentation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470766.pdf) (LWL)|ECCV|2020|[PyTorch](https://github.com/visionml/pytracking)|
|[Target-Aware Object Discovery and Association for Unsupervised Video Multi-Object Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Target-Aware_Object_Discovery_and_Association_for_Unsupervised_Video_Multi-Object_Segmentation_CVPR_2021_paper.pdf) (TAODA)|CVPR|2021||

### 4.2. Feature Matching-based VOS

table

### 4.3. Graph Optimisation-based VOS

table

### 4.4. Optical Flow-based VOS

table

### 4.5. Mask Propagation-based VOS

table

### 4.6. Long-term Propagation-based VOS

table
