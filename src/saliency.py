#!/usr/bin/env python3

import sys
import torch
from torchvision import transforms
from PIL import Image
from pprint import pprint
from matplotlib import pyplot as plt

from lib.utils import Config, load_model

def run(image_path, conf):
    device = conf.model.device     
    model = load_model(conf)

    IMAGE_SIZE = conf.model.size[0]
    
    # Open the image file
    image = Image.open(image_path)

    # Set up the transformations
    transform_ = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),            
            transforms.ToTensor(),
    ])

    # Transforms the image
    image = transform_(image)

    # Reshape the image (because the model use 
    # 4-dimensional tensor (batch_size, channel, width, height))
    image = image.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Set the device for the image
    image = image.to(device)

    # Set the requires_grad_ to the image for retrieving gradients
    image.requires_grad_()

    # Retrieve output from the image
    output, _ = model(image)

    # Catch the output
    output_idx = output.argmax()
    output_max = output[0, output_idx]

    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()

    # Retireve the saliency map and also pick the maximum value from channels on each pixel.
    # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
    saliency, _ = torch.max(image.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(IMAGE_SIZE, IMAGE_SIZE)

    # Reshape the image
    image = image.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('The Image and Its Saliency Map')
    plt.draw()
    plt.show(block=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nError: Path[s] of configuration file[s] needed.")
        print("\nUsage: ./train.py <configuration_file>\n")        
        exit(1)
    
    config = sys.argv[1]
    
    # print(f'Loading configuration "{config}"')
    config = Config.load_json(config)
    # print("Configuration")
    pprint(config)    
    
    config.model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use 1st GPU    

    image_path = "../input/processed/test/現生サンプル_コムギ_繊維_46.jpg"
    run(image_path, config)