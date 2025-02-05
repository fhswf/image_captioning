import sys
import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from data_loader import MyDataLoader
from model import EncoderCNN, DecoderRNN


class Trainer:
    """The Trainer encapsulates the model training process."""

    def __init__(self, train_loader, val_loader, encoder, decoder, optimizer, criterion=None, start_epoch=0, rounds=1):
        """Initialize the Trainer state. This includes loading the model data if start_epoch > 0."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.encoder = encoder
        self.decoder = decoder
        self.epoch = start_epoch
        self.rounds = rounds
        self.current_state_file = os.path.join('./models', 'current-model.pkl')
        self.optimizer = optimizer
        self.vocab = self.train_loader.vocab
        self.vocab_size = len(self.vocab)
        if criterion is None:
            pad_idx = self.vocab.word2idx['<pad>']
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        else: 
            self.criterion = criterion
        
        if torch.cuda.is_available():
            self.map_location = torch.device('cuda')
            self.criterion.cuda()
            self.encoder.cuda()
            self.decoder.cuda()
        else:
            self.map_location = torch.device('cpu')

        self.cider = []

        if self.epoch > 0:
            self.load()
    
    def load(self):
        """Load the model output of an epoch."""
        checkpoint = torch.load(self.current_state_file, map_location=self.map_location)

        # Load the pre-trained weights
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        self.cider = checkpoint['cider']
        print('Successfully loaded epoch {}'.format(self.epoch))

    def save_as(self, file_name):
        """Save the training state in a pickle file.

        The following values are saved: 
        - encoder parameter, 
        - decoder parameters,
        - optimizer state, 
        - current epoch,
        - list of CIDEr scores from the evaluation of past epochs.
        
        Parameters
        ----------
        file_name : str
            Name of the file to save.
        """
        torch.save({"encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "cider": self.cider,
                    "epoch": self.epoch
                   }, file_name)

    def save(self):
        """Save the training state in a pickle file.
        
        The following values are saved: 
        - encoder parameter, 
        - decoder parameters,
        - optimizer state, 
        - current epoch,
        - list of CIDEr scores from the evaluation of past epochs.
        """
        self.save_as(os.path.join("./models", "current-model.pkl"))
        self.save_as(os.path.join("./models", "epoch-model-{}.pkl".format(self.epoch)))

    def clean_sentence(self, word_idx_list):
        """Take a list of word ids and a vocabulary from a dataset as inputs
        and return the corresponding sentence as a single Python string.

        Parameters
        ----------
        word_idx_list : list
            List of word indices, i.e. embedded words.
        """
        sentence = []
        for i in range(len(word_idx_list)):
            vocab_id = word_idx_list[i]
            word = self.vocab.idx2word[vocab_id]
            if word == self.vocab.end_word:
                break
            if word != self.vocab.start_word:
                sentence.append(word)
        sentence = " ".join(sentence)
        return sentence

    def train(self):
        """Train the model for one epoch using the provided parameters. Return the epoch's average train loss."""

        # Switch to train mode
        self.encoder.train()
        self.decoder.train()

        # Keep track of train loss
        total_loss = 0

        # Start time for every 100 steps
        start_train_time = time.time()
        i_step = 0
    
        # Obtain the batch
        pbar = tqdm(self.train_loader)
        pbar.set_description('training epoch {}'.format(self.epoch))
        for batch in pbar:
            i_step += 1
            images, captions, lengths = batch[0], batch[1], batch[2]
             
            # Move to GPU if CUDA is available
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                lengths = lengths.cuda()

            # Pass the inputs through the CNN-RNN model
            features = self.encoder(images)
            outputs = self.decoder(features, captions, lengths)

            # Calculate the batch loss
            # Flatten batch dimension
            outputs = outputs.view(-1, vocab_size)
            captions = captions.view(-1)

            loss = self.criterion(outputs, captions)

            # Zero the gradients. Since the backward() function accumulates 
            # gradients, and we don’t want to mix up gradients between minibatches,
            # we have to zero them out at the start of a new minibatch
            self.optimizer.zero_grad()
            # Backward pass to calculate the weight gradients
            loss.backward()
            # Update the parameters in the optimizer
            self.optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(last=loss.item(), avg=total_loss/i_step)
            
        self.epoch += 1
        self.save()

        return total_loss / i_step

    def evaluate(self):
        """Evaluate the model for one epoch using the provided parameters. 
           Return the epoch's average CIDEr score."""

        # Switch to validation mode
        self.encoder.eval()
        self.decoder.eval()

        cocoRes = COCO()
        anns = []

        # Disable gradient calculation because we are in inference mode
        with torch.no_grad():
            pbar = tqdm(self.val_loader)
            pbar.set_description('evaluating epoch {}'.format(self.epoch));
            for batch in pbar:
                images, img_id = batch[0], batch[3]

                # Move to GPU if CUDA is available
                if torch.cuda.is_available():
                    images = images.cuda()

                # Pass the inputs through the CNN-RNN model
                features = encoder(images).unsqueeze(1)
                for i in range(img_id.size()[0]):
                    slice = features[i].unsqueeze(0)
                    outputs = decoder.sample_beam_search(slice)
                    sentence = self.clean_sentence(outputs[0])
                    id = img_id[i].item()
                    #print('id: {}, cap: {}'.format(id, sentence))
                    anns.append({'image_id': id, 'caption': sentence})
             
        for id, ann in enumerate(anns):
            ann['id'] = id
    
        cocoRes.dataset['annotations'] = anns
        cocoRes.createIndex()

        cocoEval = COCOEvalCap(self.val_loader.coco_dataset.coco, cocoRes)
        imgIds = set([ann['image_id'] for ann in cocoRes.dataset['annotations']])
        cocoEval.params['image_id'] = imgIds
        cocoEval.evaluate()
        cider = cocoEval.eval['CIDEr']
        old_max = 0
        if len(self.cider) > 0:
            old_max = max(self.cider)

        if len(self.cider) < self.epoch:
            self.cider.append(cider)
        else:
            self.cider[self.epoch-1] = cider
        self.save()
        print("DEBUG: self.epoch: {}, self.cider: {}".format(self.epoch, self.cider))
        if cider > old_max:
            print('CIDEr improved: {:.2f} => {:.2f}'.format(old_max, cider))
            self.save_as(os.path.join("./models", "best-model.pkl"))

        return self.cider[self.epoch-1]


# Set values for the training variables
batch_size = 128        # batch size
vocab_threshold = 5     # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 512        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
num_epochs = 10         # number of training epochs

# Define a transform to pre-process the training images
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Build data train_loader, applying the transforms
train_loader = MyDataLoader(transform=transform_train,
                            mode='train',
                            batch_size=batch_size,
                            vocab_threshold=vocab_threshold,
                            vocab_from_file=vocab_from_file)

transform_val = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

val_loader = MyDataLoader(transform=transform_val,
                        mode='val',
                        batch_size=batch_size,
                        vocab_threshold=vocab_threshold,
                        vocab_from_file=True)

# The size of the vocabulary
vocab_size = len(train_loader.vocab)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Specify the learnable parameters of the model
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters()) + list(encoder.resnet.parameters())

# Define the optimizer
optimizer = torch.optim.AdamW(params=params, lr=0.001, weight_decay=0.05, amsgrad=True)

trainer = Trainer(train_loader, val_loader, encoder, decoder, optimizer)

if not os.path.exists(trainer.current_state_file):
    trainer.train()
trainer.load()

# if cider is missing for current epoch, evaluater first
if len(trainer.cider) < trainer.epoch:
    print('Epoch {} not yet evaluated'.format(trainer.epoch))
    trainer.evaluate()

for i in range(num_epochs):
    trainer.train()
    trainer.evaluate()