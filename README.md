# SAMS_VQA

## Description 

This is project SAMS_VQA developed by the team composed of Arda Bati, Marjan Emadi, So Sasaki, and Sina Shahsavari.

The repository mainly consists of two implementations of Visual Question Answering (VQA). 

- The experiment1 is an implementation for Bottom-Up and Top-Down Attention for VQA, which our final report is based on. 
- The experiment2 is a different implementation of the vanilla VQA Model. The details are on the README file in each experiment directory.  

## Requirements and Usage

### Experiment1

The experiment1 requires 64G memory. To get sufficient resource on UCSD's ing6 server, create a pod as follows:
```
launch-pytorch-gpu.sh -m 64
```

#### Requirements:

```
python 2.7 
pillow
cuda 8.0
pytorch 0.3.0
```

#### Environment creation:

```
conda create -n envname python=2.7 mkl=2018 pytorch=0.3.0 -c pytorch -c intel
source activate envname
pip install --user pillow h5py
```

#### Training the Model

To train the model, run the following commands:

```
cd experiment1
sh tools/download.sh
sh tools/process.sh
python main.py
```

#### Demonstration

For the demonstration, you can the experiment results stored in our_answers.dms. This file is uploaded in experiment1/demo, but you can also generate it as follows:

```
cd experiment1/demo
python demo.py
```

Finally, you can run the demo script on Jupyter Notebook:

```
experiment1/demo/Demo.ipyenb
```

### Experiment2

#### Requirements

```
python 3.7
torchtext
tensorboardX
utils. 
```

#### Training the Model

To execute the code, run the following:

```
cd experiment2
pip install --user -r requirements.txt
mkdir results
mkdir preprocessed
mkdir preprocessed/img_vgg16feature_train
mkdir preprocessed/img_vgg16feature_val
python main.py -c config.yml
```

To skip preprocessing after the first execution, disable 'preprocess' in the config file.

The experiment2 does not include demo scripts or trained model parameters.

## Code organization 

### experiment1

Please refer to the experiment1 folder.

### experiment2

Please refer to the experiment2 folder.

### Miscellaneous scripts

 - misc: Miscellaneous scripts
 - misc/data_format_check.ipynb: Script for preliminary data visualization 
 - misc/some_useful_codes: Scripts which were not used after all

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
