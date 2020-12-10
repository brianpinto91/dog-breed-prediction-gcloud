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

# local directory to read training data. This will be used to copy from gcloud bucket if training on cloud
DATA_DIR = "./data"

# local directory to save model and log files temporarily before sending to gcloud buckets
MODEL_DIR = "./models"
LOG_DIR = "./logs"


class ImageDataset(Dataset):
    """A custom pytorch dataset class to represent image datasets

        Args:
            h5_file_path (str): The path of the h5 file which represents serialized images and labels
            aug_images (bool): whether or not to transform the images while loading. For test images
                                this should be false.
    """
    
    def __init__(self, h5_file_path, aug_images):
        h5_file = h5py.File(h5_file_path, 'r', libver='latest', swmr=True)
        self.labels = h5_file['labels'][:]
        self.features = h5_file['data'][:]
        self.aug_images = aug_images
        self.transformer = self.image_transformer()
        h5_file.close()

    def image_transformer(self):
        if self.aug_images:
            transform_list = [transforms.RandomPerspective(distortion_scale=0.3, p=0.2, fill=0),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(brightness=0.4, contrast=0.4),
                              transforms.RandomRotation(30),
                              transforms.RandomResizedCrop((224,224))]
            
            transformer = transforms.Compose([transforms.RandomApply(transform_list, p=0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.476, 0.452, 0.392],
                                                                  std=[0.235, 0.231, 0.229])
            ])
        else:
            transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.476, 0.452, 0.392],
                                     std=[0.235, 0.231, 0.229])
            ])
        return transformer

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image_i = Image.fromarray(self.features[idx])
        image_i = self.transformer(image_i)
        label_i = int(self.labels[idx])
        return image_i, label_i


def image_loader(dataset, batch_size, shuffle=True):
    """Function to return a pytorch dataloader

        Args:
            dataset (torch dataset): a pytorch dataset which is to be loaded using the dataloader
            batch_size (int): required batch size
            shuffle (bool): whether to shuffle the data during loading (default: false)
        
        Returns:
            dataloader (torch dataloader object)
    """

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_model(pretrained=True):
    """Function to obtain a pytorch resnet50 model with modified final layer

        Args:
            pretrained (bool): whether to use pretrained weights of the resnet50 model
        
        Returns:
            model (pytorch model): a pytorch model with resnet50 architecture with modeified final layer
    """

    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    in_fea = model.fc.in_features
    model.fc = nn.Linear(in_fea, 120)
    return model


def fetch_data_gcloud(data_directory):
    """Function to fetch training and testing data from gcloud bucket to the local directory of the container instance

        Args:
            data_directory (str): gcloud bucket with directory path
        
        Returns:
            None
    """

    try:
        subprocess.check_call(['/root/tools/google-cloud-sdk/bin/gsutil', '-m', 'cp', os.path.join(data_directory, "*"), DATA_DIR])
    except Exception as e:
        print(e)
        print("Could not fetch the data from the gcloud bucket. Check Check if the cloud bucket and path exist!")


def get_data_paths(data_directory):
    """Function to get the paths of training and testing data either from gcloud or local directory
        as specified. for cloud training this function will copy the data from gcloud bucket to the
        local container instances' directory and returns the paths.

        Args:
            data_directory (str): either local directory path or glcoud bucket directory path containing
                                  the training and testing files
        
        Returns:
            train_file_path (str):
            test_file_path (str):
    """

    if args.data_dir[0:2] == "gs":
        fetch_data_gcloud(args.data_dir)
    train_file_path = os.path.join(DATA_DIR, 'train.h5')
    test_file_path = os.path.join(DATA_DIR, 'test.h5')
    return train_file_path, test_file_path

def save_data_gcloud(from_path, gcloud_dest_path):
    """Function to copy local data to a gcloud bucket

        Args:
            from_path (str): local source path
            gcloud_dest_path (str): gcloud bucket and directory as destination path 
    """
    # specify full location of the gsutil installed path in container to make the code secure 
    subprocess.check_call(['/root/tools/google-cloud-sdk/bin/gsutil', 'cp', from_path,
                           gcloud_dest_path])

def save_model(save_path, model):
    """Function to save the model weights. During training on glcoud, the model is first saved in the
        container instance and then exported to gcloud

        Args:
            save_path (str): path where the model is to be saved
            model (torch model): a pytorch model which is to be saved
    """

    model_name = "torch_model"
    local_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model.state_dict(), local_save_path)
    if args.model_dir[0:2] == "gs":
        try:
            save_data_gcloud(local_save_path,
                             os.path.join(save_path, model_name))
        except Exception as e:
            print(e)
            print("Could not save the model to the cloud. Check if the cloud bucket and path exist!")


def save_job_log(save_path, best_performance_metrics, log_df):
    """Function to save the training logs and metadata. During training on glcoud, the logs is first saved in the
        container instance and then exported to gcloud

        Args:
            save_path (str): path where the data is to be saved
            best_performance_metrics (dict): the best epoch, its corressponding train and test loss as a dictionary
            log_df (pandas dataframe): a dataframe containing epoch-wise training and test accuracies and losses 
    """

    log_file_name = "training_result.csv"
    metadata_file_name = "metadata.json"
    
    log_save_path = os.path.join(LOG_DIR, log_file_name)
    log_df.to_csv(log_save_path, index=False)

    metadata_file_name = "metadata.json"
    metadata_save_path = os.path.join(LOG_DIR, metadata_file_name)
    metadata = vars(args)
    metadata.update(best_performance_metrics)
    with open(metadata_save_path, 'w') as outfile:
        json.dump(metadata, outfile)

    if args.log_dir[0:2] == "gs":
        try:
            save_data_gcloud(log_save_path,
                             os.path.join(args.log_dir, log_file_name))
            save_data_gcloud(metadata_save_path,
                             os.path.join(args.log_dir, metadata_file_name))
        except Exception as e:
            print(e)
            print("Could not save the log files to the cloud. Check if the cloud bucket and path exist!")