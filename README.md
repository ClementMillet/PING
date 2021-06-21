# PING

## Requirements

Pytorch, sk-learn, pandas and scikit-image are required modules to run the training.
It can be installed with `pip3 install scikit-image scikit-learn pandas`. 
Visit https://pytorch.org/ to install pytorch

To run the server, Docker and docker-compose are required.
It can be installed with `apt-get install docker docker-compose`

## Usage

For the training, just `cd training/` from the main directory and type `python3 model_training.py`.
To run the evaluation of the training : `python3 model_eval.py`
The path to the dataset must be changed, and the directories must have the following structures : 
.\
├── test\
│   ├── no_rock\
│   └── rock\
└── train\
    ├── no_rock\
    └── rock\
    
To run the web app, type `cd app/` and `docker-compose up --build`, with docker running. Then check http://localhost:5000/ on a browser.
