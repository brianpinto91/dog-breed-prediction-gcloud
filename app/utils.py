import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import numpy as np

MODEL_PATH = "./models/torch_model"

with open("./data/class_id_to_breed.json", 'r') as filehandler:
    class_id_to_breed = json.load(filehandler)

def resize_image(size, image):
    """Function to resize an image as defined. Black background is used to pad the image to maintain
        the ascpect ratio

        Args:
            size (tuple): required width and height as a tuple of integers
            image (PIL image object): image that is to be resized
        
        Returns:
            image (PIL image object): resized image
    """

    image.thumbnail(size, Image.ANTIALIAS)
    background = Image.new('RGB', size, (0, 0, 0))
    background.paste(image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2)))
    return background

def get_model_input(image):
    """Function to perform the preprocessing required so that it can be passed to the model for prediction

        Args:
            image (PIL image object): image that has to be transformed
        
        Returns:
            tensor (4D torch tensor): transformed image to torch 4D tensor (B, C, H, W)
    """

    transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.476, 0.452, 0.392],
                                                       std=[0.235, 0.231, 0.229])])
    image = transformer(image)
    image = image.unsqueeze(0)
    return image

def load_model(model_weights_path):
    """Function to load a model using the saved weights

        Args:
            model_weights_path (str): path where the model weights are saved. The weights of the model
                                      should match the model architecture
    """

    model = models.resnet50(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    in_fea = model.fc.in_features
    model.fc = nn.Linear(in_fea, 120)
    model_weights = torch.load(model_weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_weights)
    return model

def predict(processed_input):
    """Function to predict dog breed using the available model

        Args:
            processed_input (torch 4D tensor): transformed and preprocessed image
        
        Returns:
            breed_pred (str): name of the predicted breed
            class_prob (float): probability of the predicted breed
    """
    
    model.eval()
    with torch.no_grad():
        output = model(processed_input)
        all_probs = nn.functional.softmax(output, dim=-1)
        class_pred = torch.argmax(all_probs).item()
        class_prob = torch.max(all_probs).item()
        breed_pred = class_id_to_breed[str(class_pred)]
        class_prob = round(class_prob*100, 2)
    return breed_pred, class_prob

model = load_model(MODEL_PATH)