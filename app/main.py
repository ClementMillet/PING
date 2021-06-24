from flask import Flask, request, jsonify, render_template
import copy
import torch
import torch.nn as nn
from torchvision import models
from torchutils import transform_image, get_prediction
import io
import csv
from PIL import Image

app = Flask(__name__)

PATH = "model.pth"

model = models.resnet18(pretrained=True)
model_ori = copy.deepcopy(model)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
model_ori.eval()

input_size = 224

imagenet_class = []

with open('imagenet_class.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        imagenet_class.append(row)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes))
            tensor = transform_image(image, input_size)
            prediction = get_prediction(tensor, model, model_ori, imagenet_class)

            return str(prediction)
            
        except:
            return jsonify({'error': 'error during prediction'})
            
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    