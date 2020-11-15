import os
import pandas as pd
from PIL import Image

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

data_dir = "./data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
train_labels_df = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
test_labels_df = pd.read_csv(os.path.join(data_dir, "test_labels.csv"))
breeds_class_df = pd.read_csv(os.path.join(data_dir,
                              "breeds_to_class_map.csv"))

breeds_to_class = {x: y for x, y in zip(breeds_class_df.breed.values.tolist(),
                   breeds_class_df.class_id.values.tolist())}
class_to_breeds = {y: x for y, x in enumerate(breeds_to_class)}

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(225),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.476, 0.452, 0.392],
                             std=[0.235, 0.231, 0.229])
    ])


class ImageDataset(Dataset):
    def __init__(self, image_folder, labels_file, return_with_labels,
                 transformer=data_transform):
        self.image_folder = image_folder
        self.labels_file = labels_file
        self.transformer = transformer
        self.return_with_labels = return_with_labels

    def __len__(self):
        return self.labels_file.shape[0]

    def __getitem__(self, idx):
        image_name_i = self.labels_file.iloc[idx]['id'] + '.jpg'
        image_i = Image.open(os.path.join(self.image_folder, image_name_i))
        image_i = self.transformer(image_i)
        if self.return_with_labels:
            breed_i = self.labels_file.iloc[idx]['breed']
            label_i = breeds_to_class[breed_i]
            return image_name_i, image_i, label_i
        else:
            return image_name_i, image_i


def ImageLoader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def Model(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    in_fea = model.fc.in_features
    model.fc = nn.Linear(in_fea, 120)
    return model
