# Using Google Cloud to train and deploy a Dog Breed Classifier

[![GitHub repo size](https://img.shields.io/github/repo-size/brianpinto91/dog-breed-prediction-gcloud?logo=GitHub)]()
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/brianpinto91/dog-breed-prediction-gcloud/main)](https://www.codefactor.io/repository/github/brianpinto91/dog-breed-prediction-gcloud)

Using custom containers on gcloud ai-platform to train a dog breed classifier model.

## Table of Contents
* [Motivation](#motivation)
* [Implementation](#implementation)
* [Usage](#installation)
* [Technologies](#technologies)
* [License](#license)

## Motivation

>The hallmark of successful people is that they are always stretching themselves to learn new things - Carol S. Dweck

Cloud platforms offer great tools to manage end to end machine learning projects. I took this project to explore the [GoogleCloud][gcloud_link] ai-platform. I also wanted to learn [Docker][docker_link] which is a great tool to package a software and to train custom machine learning models on cloud platforms by securing all the dependencies and automating the training task. Finally, I also wanted to familiarize myself with the [GoogleCloud][gcloud_link] app engine by deploying the trained model to serve online predictions.  

## Implementation

There are two parts in this project. The first one is deploying a training job on [GoogleCloud ai-platform][gcloud_link]. And the second part is using the trained model to serve predictions by deploying it on [GoogleCloud app engine][gcloud_link]

### Model training on GoogleCloud

There are different ways to deploy a training job on GoogleCloud ai-platform. If you are using Tensorflow, scikit-learn, and XGBoost, then there are configured runtime verisons with required dependencies that can be directly used. However, if you are using Pytorch, as in my case, then [Docker][docker_link] containers can be used to define the dependencies and deploy the training job.

Depending on whether to train on cpu or gpu, I have created [Dockerfile_trainer_cpu](training/Dockerfile_trainer_cpu) and [Dockerfile_trainer_gpu](training/Dockerfile_trainer_gpu) Docker files respectively. The required python packages for model training are specified in the [training/requiremnets.txt](training/requirements.txt) file and is installed when the docker images are built.

The GoogleCloud project variables are defined in the [variables.txt](training/variables.txt) file. Each of these variables need to be exported before submitting the trianing job to GoogleCloud. 

In addition, I have created a [Makefile](training/Makefile) to ahve shortcuts for various tasks like creating GoogleCloud bucket and copying training data to it, building Docker images, pushing built Docker images to Googlecloud container registry, and submitting trianing jobs.

### App deployment on GoogleCloud

The Googlecloud App engine is used for deploying the trained model. The app uses [Flask][flask_link] and is served using the **WSGI** server [Gunicorn][gunicorn_link]. The python package dependencies are specified in [app/requirements.txt] file. The runtime python verison, the entrypoint, and the hardware instance for GoogleCloud app engine is defined in the [app.yaml](app/app.yaml) file. 

## Installation

## Technologies

[![Python: shield](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![HTML: shield](https://forthebadge.com/images/badges/uses-html.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/uses-css.svg)](https://forthebadge.com)<br/><br/>
[<img target="_blank" src="github-page/static/img/Pytorch_logo.png" alt="pytorch logo" height=40>](https://flask.palletsprojects.com/en/1.1.x/)
[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" height=50>](https://flask.palletsprojects.com/en/1.1.x/)
[<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" height=50>](https://gunicorn.org)
[<img target="_blank" src="github-page/static/img/docker.png" alt="docker logo" height=60>](https://www.docker.com)
[<img target="_blank" src="github-page/static/img/Google_Cloud_Logo.svg" alt="gcloud logo" height=40>](https://cloud.google.com)


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Copyright 2020 Brian Pinto

[gcloud_link]: https://cloud.google.com/
[docker_link]: https://www.docker.com/
[python_install_link]: https://docs.python-guide.org/starting/install3/linux/
[venv_setup_link]: https://docs.python.org/3/library/venv.html
[flask_link]: https://flask.palletsprojects.com/en/1.1.x/api/
[gunicorn_link]: https://gunicorn.org/
[heroku_link]: https://www.heroku.com/
[jinja_link]: https://jinja.palletsprojects.com/en/2.11.x/