from torchvision import transforms
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN

def evaluate(val_loader, encoder, decoder, criterion, vocab):
    """Evaluate the model for one epoch using the provided parameters. 
    Return the epoch's average validation loss and Bleu-4 score."""

    # Switch to validation mode
    encoder.eval()
    decoder.eval()

    # Keep track of validation loss 
    total_loss = 0

    # Disable gradient calculation because we are in inference mode
    with torch.no_grad():
        for batch in val_loader:
            images, captions = batch[0], batch[1]

            # Move to GPU if CUDA is available
            if torch.cuda.is_available():
                images = images.cuda()

            # Pass the inputs through the CNN-RNN model
            features = encoder(images)
            outputs = decoder.sample_beam_search(features)


            for i in range(len(outputs)):
                print('output: {}, gt: {}'.format(outputs[i], captions[i]))

    return

transform_val = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.CenterCrop(224),                      # get 224x224 crop from the center
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

batch_size = 64
vocab_threshold = 5

val_loader = get_loader(transform=transform_val,
    mode='val',
    batch_size=batch_size,
    vocab_threshold=vocab_threshold,
    vocab_from_file=True)