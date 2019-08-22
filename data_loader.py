"""Create the CoCoDataset and a DataLoader for it."""
import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
from pexel import PEXEL
import numpy as np
from tqdm import tqdm
import random
import json

class MyDataLoader(data.DataLoader):
    vocab = None

    def __init__(self,
                transform,
               mode="train",
               batch_size=1,
               vocab_threshold=None,
               vocab_file="./vocab.pkl",
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               pad_word="<pad>",
               vocab_from_file=True,
               num_workers=0):
        """Return the data loader.
        Parameters:
        transform: Image transform.
        mode: One of "train", "val" or "test".
        batch_size: Batch size (if in testing mode, must have batch_size=1).
        vocab_threshold: Minimum word count threshold.
        vocab_file: File containing the vocabulary. 
        start_word: Special word denoting sentence start.
        end_word: Special word denoting sentence end.
        unk_word: Special word denoting unknown words.
        vocab_from_file: If False, create vocab from scratch & override any 
                         existing vocab_file. If True, load vocab from from
                         existing vocab_file, if it exists.
        num_workers: Number of subprocesses to use for data loading 
        cocoapi_loc: The location of the folder containing the COCO API: 
                     https://github.com/cocodataset/cocoapi
        """
    
        assert mode in ["train", "val", "test"], "mode must be one of 'train', 'val' or 'test'."

        if self.vocab is None:
            if vocab_from_file == False: 
                assert mode == "train", "To generate vocab from captions file, must be in training mode (mode='train')."
            self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                    end_word, unk_word, pad_word, vocab_from_file)
     
        # COCO caption dataset
        self.coco_dataset = CocoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocabulary=self.vocab)

        self.pexel_dataset = PexelDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocabulary=self.vocab)

        if mode == "train":
            # Calculating overall maximum of caption length (needed for padding)
            max_length = max([self.coco_dataset.max_length, self.pexel_dataset.max_length])
            print("Maximum caption length: {}".format(max_length))
            self.coco_dataset.max_length = max_length
            self.pexel_dataset.max_length = max_length
 
        dataset = data.ConcatDataset([self.coco_dataset, self.pexel_dataset])
        super().__init__(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)

    

class PexelDataset(data.Dataset):
    """Dataset of free, captioned images from https:pexels.com"""
    
    def __init__(self, transform, mode, batch_size, vocabulary):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = vocabulary  
        self.pexel_annotations_file = os.path.normpath("/home/cgawron/pexels/pexels.json")
        self.pexel_img_folder = os.path.normpath("/home/cgawron/pexels/images")

        if self.mode == "train":
            self.pexel = PEXEL(self.pexel_annotations_file)
            self.pexel_ids = list(self.pexel.anns.keys())
            print("pexel: {} captions".format(len(self.pexel_ids)))
             
            print("Obtaining caption lengths...")
            pexel_tokens = [nltk.tokenize.word_tokenize(
                          str(self.pexel.anns[self.pexel_ids[index]]).lower())
                            for index in tqdm(np.arange(len(self.pexel_ids)))]
            self.caption_lengths = [len(token) for token in pexel_tokens]
            self.max_length = max(self.caption_lengths)

        else:
            print("This dataset only contains training images")
 
        
    def __getitem__(self, index):
        if self.mode == "train":
            ann_id = self.pexel_ids[index]
            caption = self.pexel.anns[ann_id]
            img_id = ann_id
            path = self.pexel.getImgPath(img_id)
            image = Image.open(os.path.join(self.pexel_img_folder, path)).convert("RGB")

            # Convert image to tensor and pre-process using transform            
            image = self.transform(image)

            orig = caption

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            length = len(caption)
            for i in range(len(caption), self.max_length+2):
                caption.append(self.vocab(self.vocab.pad_word))
            caption = torch.Tensor(caption).long()

            # Return pre-processed image and caption tensors
            return image, caption, length, ann_id

    def __len__(self):
        if self.mode == "train":
            return len(self.pexel_ids)
        else:
            return 0


class CocoDataset(data.Dataset):

    coco_paths = { "train": [ os.path.normpath("coco/annotations/captions_train2014.json"), os.path.normpath("coco/images/train2014/") ],
                   "val":  [ os.path.normpath("coco/annotations/captions_val2014.json"), os.path.normpath("coco/images/val2014/") ],
                   "test": [ os.path.normpath("coco/annotations/image_info_test2014.json"), os.path.normpath("coco/images/test2014/") ]
                 } 

    def __init__(self, transform, mode, batch_size, vocabulary):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = vocabulary
        self.coco_annotations_file = self.coco_paths[mode][0]
        self.coco_img_folder = self.coco_paths[mode][1]

        if self.mode == "train":
            self.coco = COCO(self.coco_annotations_file)
            self.coco_ids = list(self.coco.anns.keys())
              
            print("Obtaining caption lengths...")
            coco_tokens = [nltk.tokenize.word_tokenize(
                          str(self.coco.anns[self.coco_ids[index]]["caption"]).lower())
                            for index in tqdm(np.arange(len(self.coco_ids)))]
 
            self.caption_lengths = [len(token) for token in coco_tokens]
            self.max_length = max(self.caption_lengths)

        elif self.mode == "val":
            self.coco = COCO(self.coco_annotations_file)
            self.coco_ids = list(self.coco.imgs.keys())
 
        
    def __getitem__(self, index):
        # Obtain image and caption if in training or validation mode
        if self.mode == "train":
            ann_id = self.coco_ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]
            image = Image.open(os.path.join(self.coco_img_folder, path)).convert("RGB")

            # Convert image to tensor and pre-process using transform
            
            image = self.transform(image)

            orig = caption

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            length = len(caption)
            for i in range(len(caption), self.max_length+2):
                caption.append(self.vocab(self.vocab.pad_word))
            caption = torch.Tensor(caption).long()

            # Return pre-processed image and caption tensors
            return image, caption, length, ann_id

        elif self.mode == "val":
            img_id = self.coco_ids[index]
            path = self.coco.loadImgs(img_id)[0]["file_name"]
            image = Image.open(os.path.join(self.coco_img_folder, path)).convert("RGB")
            image = self.transform(image)

            return image, torch.empty(1), torch.empty(1), img_id

 
    def __len__(self):
        return len(self.coco_ids)