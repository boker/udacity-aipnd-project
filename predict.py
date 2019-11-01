import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import check_gpu
from torchvision import models, transforms

# ------------------------------------------------------------------------------- #
# Function Definitions
# ------------------------------------------------------------------------------- #
# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # Point towards image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to impage file for prediction.',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        default = 'checkpoint.pth')
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.',
                        default = 3)
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to real names.',
                        default = 'cat_to_name.json')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')

    # Parse args
    args = parser.parse_args()
    
    return args

# Function load_checkpoint(checkpoint_path) loads our saved deep learning model from checkpoint
def load_checkpoint(checkpoint_path):
    # Load the saved file
    checkpoint = torch.load(checkpoint_path)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.name = checkpoint['architecture']
    print(checkpoint['architecture'])
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Function process_image(image_path) performs cropping, scaling of image for our model
def process_image(image_path):
    test_image = PIL.Image.open(image_path)

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    pil_tfd = transform(test_image)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(array_im_tfd).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)
    
    return img_add_dim
#     return np_image


def predict(image_tensor, model, device, cat_to_name, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities(k), top_labels
    '''
    
    # check top_k
    if type(top_k) == type(None):
        top_k = 5
        print("Top K not specified, assuming K=5.")
    
    # Set model to evaluate
    model.eval();

    model = model.to(device)
    torch_image = image_tensor.to(device)

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    with torch.no_grad():
        log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    """
    Converts two lists into a dictionary to print on screen
    """
    
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))
    

# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    
    # Process Image
    image_tensor = process_image(args.image)
    
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model, 
                                                 device, cat_to_name,
                                                 args.top_k)
    
    # Print out probabilities
    print_probability(top_flowers, top_probs)

# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()