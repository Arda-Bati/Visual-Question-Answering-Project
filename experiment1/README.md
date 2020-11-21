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

#### Usage

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

## Code organization 

 - experiment1: An implementation for Bottom-Up and Top-Down Attention for VQA
 - experiment1/main.py: Main script to set up models and dataloaders and run training module
 - experiment1/train.py: Module for training and evaluation
 - experiment1/base_model.py: Module for BaseModel, which controls WordEmbedding, QuestionEmbedding, Attention, FCnet, and classifier
 - experiment1/language_model.py: Module for WordEmbedding and QuestionEmbedding
 - experiment1/fc.py: Module for fully-connected network part
 - experiment1/attention.py: Module for Attention
 - experiment1/dataset.py: Module for dictionary and dataloading
 - experiment1/utils.py: Module for utility scripts
 - experiment1/demo: Files related to demo
 - experiment1/demo/Demo.ipynb: Main notebook for demo 
 - experiment1/demo/demo.py: Script to generate answers
 - experiment1/demo/base_model.py: Module for BaseModel
 - experiment1/demo/glove6b_init_300d.npy: Parameters for WordEmbedding
 - experiment1/demo/model.pth: Trained model
 - experiment1/demo/our_answers.dms: Result answers for sample questions
 - experiment1/demo/test.dms: Image features for sample questions
 - experiment1/demo/trainval_label2ans.pkl: Dictionary to decode answers
 - experiment1/demo/readme.md: Other information for demo
 - experiment1/tools: Files for preprocessing and downloading
 - experiment1/tools/download.sh: Script to download raw image features
 - experiment1/tools/process.sh: Script to run create_dictionary, compute_softscore, and detection_features_converter
 - experiment1/tools/create_dictionary.py: Script to create dictionary
 - experiment1/tools/compute_softscore.py: Script to compute softscore
 - experiment1/tools/detection_features_converter.py: Script to generate detection features
 - experiment1/data: Data files
 - experiment1/data/train_ids.pkl: Indeces of training data
 - experiment1/data/val_ids.pkl: Indeces of validation data
 - experiment1/README.md: Information for experiment1

## References

 - https://arxiv.org/abs/1707.07998 
 - https://github.com/hengyuan-hu/bottom-up-attention-vqa. 
 - http://www.visualqa.org/challenge.html
 - http://www.visualqa.org/evaluation.html
 - https://arxiv.org/pdf/1611.09978.pdf

