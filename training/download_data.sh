#!/bin/bash

#download the data to training/data and name it iamges.tar
wget ‐P "./training/data" http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

#untar
tar -xf ./training/data/images.tar -C ./training/data

#delete the downloaded tar file
rm -f "./training/data/images.tar"