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

def get_loader(transform,
               mode="train",
               batch_size=1,
               vocab_threshold=None,
               vocab_file="./vocab.pkl",
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc="."):
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
    if vocab_from_file == False: 
        assert mode == "train", "To generate vocab from captions file, \
               must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file
    pexel_annotations_file = None
    pexel_img_folder = None
    if mode == "train":
        if vocab_from_file == True: 
            assert os.path.exists(vocab_file), "vocab_file does not exist.  \
                   Change vocab_from_file to False to create vocab_file."
        coco_img_folder = os.path.join(cocoapi_loc, "cocoapi/images/train2014/")
        coco_annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/captions_train2014.json")
        pexel_img_folder = os.path.normpath("/home/cgawron/pexels/images")
        pexel_annotations_file = os.path.normpath("/home/cgawron/pexels/pexels.json")
    if mode == "val":
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        coco_img_folder = os.path.join(cocoapi_loc, "cocoapi/images/val2014/")
        coco_annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/captions_val2014.json")
    if mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        coco_img_folder = os.path.join(cocoapi_loc, "cocoapi/images/test2014/")
        coco_annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/image_info_test2014.json")
    
    # COCO caption dataset
    dataset = JoinedDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          coco_annotations_file=coco_annotations_file,
                          vocab_from_file=vocab_from_file,
                          coco_img_folder=coco_img_folder,
                          pexel_annotations_file=pexel_annotations_file,
                          pexel_img_folder=pexel_img_folder)

    if mode == "train":
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

class JoinedDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, vocab_from_file, coco_annotations_file, coco_img_folder, pexel_annotations_file, pexel_img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, coco_annotations_file, pexel_annotations_file, vocab_from_file)
        self.coco_img_folder = coco_img_folder
        self.pexel_img_folder = pexel_img_folder
        if self.mode == "train" or self.mode == "val":
            self.coco = COCO(coco_annotations_file)
            self.coco_ids = list(self.coco.anns.keys())
            self.pexel = None
            self.pexel = PEXEL(pexel_annotations_file)
            self.pexel_ids = list(self.pexel.anns.keys())
            print("pexel: {} captions".format(len(self.pexel_ids)))
             
            print("Obtaining caption lengths...")
            coco_tokens = [nltk.tokenize.word_tokenize(
                          str(self.coco.anns[self.coco_ids[index]]["caption"]).lower())
                            for index in tqdm(np.arange(len(self.coco_ids)))]
            pexel_tokens = [nltk.tokenize.word_tokenize(
                          str(self.pexel.anns[self.pexel_ids[index]]).lower())
                            for index in tqdm(np.arange(len(self.pexel_ids)))]
            self.caption_lengths = [len(token) for token in coco_tokens + pexel_tokens]
        # If in test mode
        else:
            test_info = json.loads(open(coco_annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]
        
    def __getitem__(self, index):
        # Obtain image and caption if in training or validation mode
        if self.mode == "train" or self.mode == "val":
            if index >= len(self.coco_ids):
                index -= len(self.coco_ids)
                ann_id = self.pexel_ids[index]
                caption = self.pexel.anns[ann_id]
                img_id = ann_id
                path = self.pexel.getImgPath(img_id)
                image = Image.open(os.path.join(self.pexel_img_folder, path)).convert("RGB")
            else: 
                ann_id = self.coco_ids[index]
                caption = self.coco.anns[ann_id]["caption"]
                img_id = self.coco.anns[ann_id]["image_id"]
                path = self.coco.loadImgs(img_id)[0]["file_name"]
                image = Image.open(os.path.join(self.coco_img_folder, path)).convert("RGB")

            # Convert image to tensor and pre-process using transform
            
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # Return pre-processed image and caption tensors
            return image, caption

        # Obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image and pre-processed image tensor
            return orig_image, image


    def get_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == \
                               sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return len(self.coco_ids) + len(self.pexel_ids)
        else:
            return len(self.paths)
