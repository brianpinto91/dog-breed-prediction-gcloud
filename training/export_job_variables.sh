#!/bin/bash

set -o allexport

# variables to be exported
MODELS_DIR=gs://${BUCKET_NAME}/models/${IMAGE_TAG}_$(date +%Y%m%d_%H%M%S)
LOGS_DIR=gs://${BUCKET_NAME}/logs/${IMAGE_TAG}_$(date +%Y%m%d_%H%M%S)
JOB_NAME=${IMAGE_TAG}_DATE_TIME_$(date +%Y%m%d_%H%M%S)

set -o allexport