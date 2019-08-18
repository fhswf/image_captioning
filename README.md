# Image Captioning

## Introduction

Generate captions from images using a deep learning model. When given an image, the model is able to describe in English what is in the image. In order to achieve this, our model is comprised of an **encoder** which is a CNN and a **decoder** which is an RNN. The CNN encoder is given images for a classification task and its output is fed into the RNN decoder which outputs English sentences.

The model and the tuning of its hyperparamaters are based on ideas presented in the paper [**Show and Tell: A Neural Image Caption Generator**](https://arxiv.org/pdf/1411.4555.pdf) and [**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**](https://arxiv.org/pdf/1502.03044.pdf).

We use the Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) [dataset](http://cocodataset.org/#home) for this project. It is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms. For instructions on downloading the data, see the **Data** section below.

## Changes from the original project by Trang Nguyen
This project is forked from a [git repository](https://github.com/ntrang086/image_captioning) created by [Trang Nguyen](https://github.com/ntrang086).
The following points have been changed:
* The **encoder** used is a pre-trained instance of [**ResNeXt101_32x8d**](https://arxiv.org/abs/1611.05431). 
* The **decoder** used is a two-layer [**GRU**](https://en.wikipedia.org/wiki/Gated_recurrent_unit) RNN instead of a single-layer LSTM RNN.
* Training is done in `training.py`. Instead of sampling training captions into batches of fixed length, the training captions are padded and packed with 
  [`torch.nn.utils.rnn.pack_padded_sequence`](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence). This improves training speed on my machine and ensures that all training samples are used during a training epoch.
* Evaluation of the model is done with the 'official' [MS COCO Evaluation Code](https://github.com/salaniz/pycocoevalcap) and the 
  CIDEr score is used to decide whether the model has improved or not.
* Besides MS COCO, I used *free* captioned images from [pexels.com](https://www.pexels.com) for the training.
* I included a simple REST service `rest_service.py` which can be used to call the model via a simple web frontend. 
  See this [**online demo**](https://jupiter.fh-swf.de/captioning/index.html) on my home page.

## Code
*TBD*

## Setup

1. Install pycocoevalcap and the pycocotools by running 

   ```
   pip install git+https://github.com/salaniz/pycocoevalcap
   ```

2. Install PyTorch (4.0 recommended) and torchvision.
	
	```
	pip install pytorch torchvision 
	```

3. Other dependencies:

* Python 3
* nltk
* numpy
* scikit-image
* matplotlib
* tqdm

## Data

Download the following data from the [COCO website](http://cocodataset.org/#download), and place them, as instructed below, into a `coco` subdirectory located *inside* this project's directory:

* under **Annotations**, download:
  - **2014 Train/Val annotations [241MB]** (extract `captions_train2014.json`, `captions_val2014.json`, `instances_train2014.json` and `instances_val2014.json`, and place them in the subdirectory `coco/annotations/`)
  - **2014 Testing Image info [1MB]** (extract `image_info_test2014.json` and place it in the subdirectory `coco/annotations/`)
* under **Images**, download:
  - **2014 Train images [83K/13GB]** (extract the `train2014` folder and place it in the subdirectory `coco/images/`)
  - **2014 Val images [41K/6GB]** (extract the `val2014` folder and place it in the subdirectory `coco/images/`)
  - **2014 Test images [41K/6GB]** (extract the `test2014` folder and place it in the subdirectory `coco/images/`)
          
## Run

To train the model, run:

```bash
python training.py
```

To run any IPython Notebook, use:

```bash
jupyter notebook <notebook_name.ipynb>
```
