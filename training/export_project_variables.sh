#!/bin/bash

set -o allexport

# variables to be exported
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform
REGION=europe-west3
IMAGE_REPO_NAME=pytorch_trainer
IMAGE_TAG=GPU_V1
IMAGE_URI=eu.gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}
DATA_DIR=gs://${BUCKET_NAME}/data

set -o allexport