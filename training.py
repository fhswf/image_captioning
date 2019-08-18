import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from pycocotools.coco import COCO
import math
import torch.utils.data as data
import numpy as np
import requests
import time

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def train(train_loader, encoder, decoder, criterion, optimizer, vocab_size,
          epoch, start_loss=0.0):
    """Train the model for one epoch using the provided parameters. Return the epoch's average train loss."""

    # Switch to train mode
    encoder.train()
    decoder.train()

    # Keep track of train loss
    total_loss = start_loss

    # Start time for every 100 steps
    start_train_time = time.time()
    i_step = 0
    
    # Obtain the batch
    for batch in train_loader:
        i_step += 1
        images, captions, lengths = batch[0], batch[1], batch[2]
             
        # Move to GPU if CUDA is available
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()
        # Pass the inputs through the CNN-RNN model
        features = encoder(images)
        outputs = decoder(features, captions, lengths)

        #print("1: outputs: {}, captions: {}".format(outputs.size(), captions.size()))
        # Calculate the batch loss
        outputs = outputs.view(-1, vocab_size)
        captions = captions.view(-1)
        #print("2: outputs: {}, captions: {}".format(outputs.size(), captions.size()))

        loss = criterion(outputs, captions)

        # Zero the gradients. Since the backward() function accumulates 
        # gradients, and we don’t want to mix up gradients between minibatches,
        # we have to zero them out at the start of a new minibatch
        optimizer.zero_grad()
        # Backward pass to calculate the weight gradients
        loss.backward()
        # Update the parameters in the optimizer
        optimizer.step()

        total_loss += loss.item()

        # Get training statistics
        stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f" \
                % (epoch, i_step, len(train_loader), time.time() - start_train_time,
                   loss.item(), np.exp(loss.item()))
        # Print training statistics (on same line)
        print("\r" + stats, end="")
        sys.stdout.flush()

        # Print training stats (on different line), reset time and save checkpoint
        if i_step % 100 == 0:
            print("\r" + stats)
            start_train_time = time.time()
    
    filename = os.path.join("./models", "train-model-{}.pkl".format(epoch))
    save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch)
    return total_loss / i_step

def save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, train_step=1):
    """Save the following to filename at checkpoints: encoder, decoder,
    optimizer, total_loss, epoch, and train_step."""
    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "total_loss": total_loss,
                "epoch": epoch
               }, filename)

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

# Build data loader, applying the transforms
train_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file)

# The size of the vocabulary
vocab_size = len(train_loader.dataset.vocab)

# Initialize the encoder and decoder
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

# Define the loss function
pad_idx = train_loader.dataset.vocab.word2idx['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# Specify the learnable parameters of the model
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())

# Define the optimizer
optimizer = torch.optim.Adam(params=params, lr=0.001)

if torch.cuda.is_available():
    map_location=torch.device('cuda')
else:
    map_location=torch.device('cpu')
checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'), map_location=map_location)

# Load the pre-trained weights
#encoder.load_state_dict(checkpoint['encoder'])
#decoder.load_state_dict(checkpoint['decoder'])
#epoch = checkpoint['epoch']

epoch = 1

train_loss = train(train_loader, encoder, decoder, criterion, optimizer, 
                   vocab_size, epoch)