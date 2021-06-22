from torchutils import transform_image, get_prediction
from PIL import Image

PATH = "/mnt/d/Users/cleme/Pictures/hkmeme.png"

im = Image.open(PATH)
tensor = transform_image(im)
prediction = get_prediction(tensor)

print(prediction)