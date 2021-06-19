from flask import Flask, request, jsonify, render_template

from torchutils import transform_image, get_prediction
import io
from PIL import Image

app = Flask(__name__)

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
            tensor = transform_image(image)
            prediction = get_prediction(tensor)
            
            '''
            flag = lambda x : True if (x.item() == 1) else False
            return str(flag(prediction))
            '''
            
            #class_name = lambda x : "a rock" if (x.item() == 1) else "not a rock" 
            #data = {'prediction': prediction}
            return str(prediction)
            
            disp_answer = lambda x : render_template("rock.html") if (x.item() == 1) else render_template("no_rock.html")
            return disp_answer(prediction)
           
            
        except:
            return jsonify({'error': 'error during prediction'})
            
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')