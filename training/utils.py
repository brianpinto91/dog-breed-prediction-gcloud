import os
import subprocess
import pandas as pd
from PIL import Image
import h5py
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.476, 0.452, 0.392],
                             std=[0.235, 0.231, 0.229])
    ])


class ImageDataset(Dataset):
    def __init__(self, h5_file_path,
                 transformer=data_transform):
        h5_file = h5py.File(h5_file_path, 'r', libver='latest', swmr=True)
        self.labels = h5_file['labels'][:]
        self.features = h5_file['data'][:]
        h5_file.close()
        self.transformer = data_transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image_i = Image.fromarray(self.features[idx])
        image_i = self.transformer(image_i)
        label_i = int(self.labels[idx])
        return image_i, label_i


def ImageLoader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def Model(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    in_fea = model.fc.in_features
    model.fc = nn.Linear(in_fea, 120)
    return model


def get_data_paths(args):
    if args.model_dir[0:2] == "gs":
        copy_gcloud_data(args)
    train_file_path = os.path.join("../data", 'train.h5')
    test_file_path = os.path.join("../data", 'test.h5')
    return train_file_path, test_file_path

def copy_gcloud_data(args):
    try:
        subprocess.check_call(['gsutil', 'cp', os.path.join(args.data_dir, "*"), "../data"])
    except Exception as e:
        print(e)
        print("Could not fetch the data from the gcloud bucket. Check Check if the cloud bucket and path exist!")

def save_model(args, model):
    model_name = "torch_model"
    local_save_path = os.path.join("../models", model_name)
    torch.save(model.state_dict(), local_save_path)
    if args.model_dir[0:2] == "gs":
        try:
            subprocess.check_call(['gsutil', 'cp', local_save_path,
                                  os.path.join(args.model_dir, model_name)])
        except Exception as e:
            print(e)
            print("Could not save the model to the cloud. Check if the cloud bucket and path exist!")


def save_job_log(args, log_df):
    log_file_name = "training_result.csv"
    local_save_path = os.path.join("../logs", log_file_name)
    log_df.to_csv(local_save_path, index=False)
    if args.log_dir[0:2] == "gs":
        try:
            subprocess.check_call(['gsutil', 'cp', local_save_path,
                                  os.path.join(args.log_dir, log_file_name)])
        except Exception as e:
            print(e)
            print("Could not save the log file to the cloud. Check if the cloud bucket and path exist!")


def save_train_metadata(args, best_performance_metrics):
    metadata_file_name = "metadata.txt"
    local_save_path = os.path.join("../logs", metadata_file_name)
    json_args = json.dumps(vars(args))
    with open(local_save_path, 'w') as outfile:
        json.dump(json_args, outfile)
    if args.log_dir[0:2] == "gs":
        try:
            subprocess.check_call(['gsutil', 'cp', local_save_path,
                                  os.path.join(args.log_dir, metadata_file_name)])
        except Exception as e:
            print(e)
            print("Could not save the metadata file to the cloud. Check if the cloud bucket and path exist!")
