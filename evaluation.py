from torchvision import transforms
import torch
import torch.nn as nn
import os
import io
from tqdm import tqdm
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from utils import clean_sentence
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from collections import namedtuple


def evaluate(loader, encoder, decoder, criterion, vocab):
    """Evaluate the model for one epoch using the provided parameters. 
    Return the epoch's average validation loss and Bleu-4 score."""

    # Switch to validation mode
    encoder.eval()
    decoder.eval()

    cocoRes = COCO()
    anns = []

    # Disable gradient calculation because we are in inference mode
    with torch.no_grad():
        for batch in tqdm(loader):
            images, img_id = batch[0], batch[1]

            # Move to GPU if CUDA is available
            if torch.cuda.is_available():
                images = images.cuda()

            # Pass the inputs through the CNN-RNN model
            features = encoder(images).unsqueeze(1)
            for i in range(img_id.size()[0]):
                slice = features[i].unsqueeze(0)
                outputs = decoder.sample_beam_search(slice)
                sentence = clean_sentence(outputs[0], vocab)
                id = img_id[i].item()
                #print('id: {}, cap: {}'.format(id, sentence))
                anns.append({'image_id': id, 'caption': sentence})
             
    for id, ann in enumerate(anns):
        ann['id'] = id
    
    cocoRes.dataset['annotations'] = anns
    cocoRes.createIndex()
    return cocoRes

transform_val = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

batch_size = 128
vocab_threshold = 5
embed_size = 512        # dimensionality of image and word embeddings
hidden_size = 512       # number of features in hidden state of the RNN decoder
num_epochs = 10          # number of training epochs

annFile = "coco/annotations/captions_val2014.json"

coco = COCO(annFile)


loader = get_loader(transform=transform_val,
    mode='val',
    batch_size=batch_size,
    vocab_threshold=vocab_threshold,
    vocab_from_file=True)
vocab_size = len(loader.dataset.vocab)
print('vocabulary size: {}'.format(vocab_size))

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
if torch.cuda.is_available():
    decoder.cuda()
    encoder.cuda()
    

criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
if torch.cuda.is_available():
    map_location=torch.device('cuda')
else:
    map_location=torch.device('cpu')
checkpoint = torch.load(os.path.join('./models', 'best-model.pkl'), map_location=map_location)



# Load the pre-trained weights
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

# Create captions for validation images
cocoRes = evaluate(loader, encoder, decoder, criterion, loader.dataset.vocab)

cocoEval = COCOEvalCap(coco, cocoRes)
imgIds = set([ann['image_id'] for ann in cocoRes.dataset['annotations']])
cocoEval.params['image_id'] = imgIds
cocoEval.evaluate()

def getCIDEr(ann):
    return ann['cider']

anns = cocoRes.dataset['annotations']
for ann in anns:
    ann['cider'] = cocoEval.imgToEval[ann['image_id']]['CIDEr']

anns.sort(key = getCIDEr)
result = { 'eval': cocoEval.eval, 'annotations': anns}

f = open('result.json', 'wt')
json.dump(result, f, indent=1)
f.close()