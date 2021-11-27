# Few-shot Classification for Web Service

This repository contains the source code and datasets for our FGCS paper: *Multi-Information Fusion Based Few-shot Web Service Classification*

The code is partially referred to https://github.com/YujiaBao/Distributional-Signatures

A few-shot Web service classification method called MIF-FWSC (multi-information fusion based few-shot Web service classification) based on meta-learning framework is proposed to classify the categories with only a few Web services. MIF-FWSC can find key words in the service description and exploit the knowledge in head categories to make the model more robust across different categories. It also incorporates the information contained in the category names to enhance the classification ability. 

A series of experiments are conducted on two real-world datasets to demonstrate the effectiveness of our proposed model. The results show that MIF-FWSC achieves state-of-the-art performance for few-shot Web service classification.

# Data

ProgrammableWeb (https://www.programmableweb.com) and Amazon Web service marketplace (https://aws.amazon.com/marketplace) are by far the two largest online Web service registries. New Web services and new service categories are continuously registered in them. We collected a dataset from ProgrammableWeb on Jan 10, 2020 and a dataset from Amazon Web service marketplace on Aug 7, 2021. All datasets are in the directory data.

# Environment

- NVIDIA 2080
- CUDA 11.1

## Dependencies

- **Python 3** (tested on python 3.8)
- [PyTorch](https://github.com/pytorch/pytorch) 1.9.0
    - with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow)
- [numpy](https://www.numpy.org) 1.18.5
- [torchtext](https://github.com/pytorch/text) 0.5.0
- [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) 1.1.0
- [termcolor](http://pypi.python.org/pypi/termcolor) 1.1.0
- [tqdm](https://github.com/tqdm/tqdm) 4.50.2
- [pandas](https://pandas.pydata.org) 1.1.3
- [nltk](http://nltk.org/) 3.5
- [gensim](http://radimrehurek.com/gensim) 3.8.3
- [joblib](https://joblib.readthedocs.io) 0.17.0

You can use the python package manager of your choice (*pip/conda*) to install the dependencies.
The code is tested on the *Linux* operating system.

## Usage

1. Download FASTTEXT pre-trained word embeddings from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec). Then put it into the directory **cache**.
2. Train **word2vec** embeddings on the Web service dataset based on the pre-trained FastText word embeddings
```
python w2v.py --dataset=[pw/aws]
```
3. Modify run.sh and run it.
```
bash run.sh
```
