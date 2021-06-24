# PING

## Goal

The goal of this project is to create a web service based on a classifier, which can detect the presence of rock in an image.

The project contains a python script for training using finetuning on the resnet model, and another to evaluate the model.\
The web service use flask, and is containerized via docker.

The web app takes an image and returns a str object, with the label 'rock' if there the classifier detect the presence of a rock in an image, else it returns the name of the imagenet class that the base resnet model has predicted.

## Requirements

Pytorch, sk-learn, pandas and scikit-image are required modules to run the training.\
It can be installed with `pip3 install scikit-image scikit-learn pandas`.\
Visit https://pytorch.org/ to install pytorch.

To run the server, Docker and docker-compose are required.\
It can be installed with `apt-get install docker.io docker-compose`.

## Usage

For the training, just `cd training/` from the main directory and type `python3 model_training.py`.\
To run the evaluation of the training : `python3 model_eval.py`.\
The path to the dataset must be changed, and the directories must have the following structure :
```
.
├── test
│   ├── no_rock
│   └── rock
└── train
    ├── no_rock
    └── rock
```
To run the web app, type `cd app/` and `docker-compose up --build`, with docker running. Then check http://localhost:5000/ on a browser.
