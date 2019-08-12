import os
from pycocotools.coco import COCO
from torchvision import transforms
import torch
import numpy as np
from vocabulary import Vocabulary

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import clean_sentence, get_prediction

class Annotator():
    def __init__(self):
        self.transform = transforms.Compose([ 
            transforms.Resize(256),                          # smaller edge of image resized to 256
            transforms.CenterCrop(224),                      # get 224x224 crop from the center
            transforms.ToTensor(),                           # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))])
        
        # Load cherckpoint with best model
        self.checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'), 'cpu')
        # Specify values for embed_size and hidden_size - we use the same values as in training step
        self.embed_size = 512
        self.hidden_size = 512

        # Get the vocabulary and its size
        self.vocab = Vocabulary(None, './vocab.pkl', "<start>", "<end>", "<unk>", "", True)
        self.vocab_size = len(self.vocab)

        # Initialize the encoder and decoder, and set each to inference mode
        self.encoder = EncoderCNN(self.embed_size)
        self.encoder.eval()
        self.decoder = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size)
        self.decoder.eval()

        # Load the pre-trained weights
        self.encoder.load_state_dict(self.checkpoint['encoder'])
        self.decoder.load_state_dict(self.checkpoint['decoder'])

        # Move models to GPU if CUDA is available.
        #if torch.cuda.is_available():
        #   encoder.cuda()
        #   decoder.cuda()

    def annotate(self, image):
        transformed = self.transform(image).unsqueeze(0)
        features = self.encoder(transformed).unsqueeze(1)

        # Pass the embedded image features through the model to get a predicted caption.
        output = self.decoder.sample_beam_search(features)
        print('example output:', output)
        sentence = clean_sentence(output[0], self.vocab)
        print('example sentence:', sentence)
        return sentence
