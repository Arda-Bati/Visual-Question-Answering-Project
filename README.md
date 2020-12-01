# SAMS_VQA

## Introduction

This is project SAMS_VQA developed by the team composed of Arda Bati, Marjan Emadi, So Sasaki, and Sina Shahsavari.

The repository mainly consists of two implementations of Visual Question Answering (VQA). 

- The experiment1 is an implementation for Bottom-Up and Top-Down Attention for VQA, which our final report is based on. 
- The experiment2 is a different implementation of the vanilla VQA Model. The details are on the README file in each experiment directory.  

## Requirements and Usage

### Experiment1

Please refer to the experiment1 folder.

### Experiment2

Please refer to the experiment2 folder.

### Miscellaneous scripts

 - misc: Miscellaneous scripts
 - misc/data_format_check.ipynb: Script for preliminary data visualization 
 - misc/some_useful_codes: Scripts which were not used after all

## Description

### VQA Concept

Visual Question Answering (VQA) Challenge has attracted much interest since its inception in
2015. The system of the VQA challenge takes input images and free-form, open-ended questions that
are of natural language format. The aim of the challenge is to produce accurate, open-ended answers
to the questions. 

The challenge combines the disciplines of Natural Language Processing (NLP) and
Computer Vision. Deep Neural Networks typically form the backbone of the system. The NLP role
is usually filled by LSTMs or other RNN variants, and the Computer Vision role is filled by popular
image classifiers/feature extractors such as VGGNet or ResNet.

<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/vqa0.jpg" width="600"/>

### VQA Network Structure

In general, for the Baseline VQA, we need to develop 2 channels [1]:

1.Image channel  creates an embedding for the visual part, and extracts the features out of them. In the
baseline model (experiment 2) we used method 1. For the baseline we used VGGNet. The activations from the last hidden layer of VGGNet are used as 4096-dim image embedding.

2.Question Channel provides an embedding for the questions. We experiment with the LSTM Q (experiment 1) An LSTM with one hidden layer is used to obtain 1024-dim embedding for the question. The embedding
obtained from the LSTM is a concatenation of last cell state and last hidden state representations
(each being 512-dim) from the hidden layer of the LSTM. Each question word is encoded with
300-dim embedding by a fully-connected layer + tanh non-linearity which is then fed to the LSTM.
The input vocabulary to the embedding layer consists of all the question words seen in the training
dataset.

<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/vqa1.png" width="600"/>

### Attention Mechanism

The mechanism of attention is used in Neural Networks to focus on specific parts or subsets of the
input, giving less priority to the rest. Visual attention is a subset of attention, in which the network
focuses on specific areas of an image. Visual attention can be considered in two main ways, Hard
and Soft Attention. Briefly, Hard Attention multiplies features of the image by a mask composed
of 0s and 1s. Parts of the image are either included with full intensity or completely excluded. This
is similar to cropping the image, where the cropped part(s) can be of any shape. Soft Attention, on
the other hand, uses a mask of values between 0 and 1 (or possibly other ranges, but not just 0 or
1). In this case, the image is not actually cropped, but the intensity of different parts is changed. An
example of Soft Attention can be seen below (in the context of image captioning, but it can also be
considered for VQA).


<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/vqa2.png" width="600"/>

   
### Main Method

There are already several Top Down methods that are used for image captioning and VQA as stated in
[2]. The novelty here comes from the combination of Top Down and Bottom Up attention mechanisms.
As explained before, the Bottom Up Model isolates proposal regions from the image that possibly
contain objects. Then, the Top Down Attention mechanism assigns weights to each of these regions,
considering the question asked about the image.

<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/bottom_up.png" width="600"/>

Similar to the other methods that are used on the VQA Dataset [10], the architecture has two main
components. These are the processing of the question and the image. For the question, firstly each
word the question contains is transformed to 300 dimension word embeddings. The word embedding
is initialized by pre-trained GloVe vectors introduced in [11]. After this step, the embedded words are
put through a Gated Recurrent Unit (GRU) [12]. Each question is represented by a hidden state of the
GRU (with dimension is 512). The encoded question is fed both to the Top Down (Soft) Attention
mechanism and the joint multimodal embedding of the question and the image.

   
<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/method.png" width="600"/>

## Modification

### Feature Extraction and Pre-training

- (1) Visual Genome dataset is not used for the pre-training of the Bottom Up Attention Model. The
original pre-training complicates the whole process. However, the final accuracy results do not seem
to be affected by this pre-training, hence it was removed. 
- (2) Keeping the number of proposed regions
in the RPN fixed at (K=36). This decision comes from the original paper which mentions that using
the top 36 proposed regions for each image worked quite well for their purposes.

### Network Changes

- (1) Instead of a two-stream classifier which classifies object and attribute classes, a single stream
classifier is used, which does not have pre-training. 
- (2) Instead of the gated tanh activations, easier
ReLU activations are used. 
- (3) The number of neurons in the layers is doubled throughout the
architecture. This alleviates the impact of some of the modifications.

### Training Changes

- (1) Dropout is added during training, which makes the network better resist overfitting. 
- (2) Batch Normalization is used during training. 
- (3) Instead of the AdaDelta optimizer Adamax [16] is used as
the baseline. However, we try additional optimizers in our experiments. 
- (4) As ReLU activations are being used, Gradient Clipping is added to prevent exploding gradients.

### Modified Attention

- (1) The concentration based attention in the top-down attention model is changed by a projection
based model. This new attention model is a modified version of the one described in the paper [17].
This model focuses on the relationships between the attended objects / regions.

## Results

todo...

<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/results1.png" width="600"/>
<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/results2.png" width="600"/>
<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/results3.png" width="600"/>

### Success and Failure Examples

<img src="https://github.com/Arda-Bati/Visual-Question-Answering-Project/blob/master/images/succ_fail.png" width="600"/>

## References

### Experiment1

 - https://arxiv.org/abs/1707.07998 
 - https://github.com/hengyuan-hu/bottom-up-attention-vqa. 
 - http://www.visualqa.org/challenge.html
 - http://www.visualqa.org/evaluation.html
 - https://arxiv.org/pdf/1611.09978.pdf

### Experiment2

 - https://github.com/Shivanshu-Gupta/Visual-Question-Answering
 - https://vqa.cloudcv.org/
 - https://arxiv.org/abs/1505.00468
 - https://arxiv.org/pdf/1511.02274
 - https://arxiv.org/abs/1705.06676
 - http://visualqa.org/download.html
 - https://github.com/Shivanshu-Gupta/Visual-Question-Answering/config
