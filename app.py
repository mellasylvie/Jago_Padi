from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('model_padi.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(160,160))
    im_array = np.asarray(image)
    im_array = im_array*(1/255)
    im_input = tf.reshape(im_array, shape = [1, 160, 160, 3])
    pred = np.argmax(model.predict(im_input))
    predict_array = model.predict(im_input)[0]
    percent = (np.max(predict_array)*100).round(2)

    if pred==0:
        desc = 'Brownspot'
    elif pred==1:
        desc = 'Healthy'
    elif pred==2:
        desc = 'Hispa'
    elif pred==3:
        desc = 'LeafBlast'

    classification = '%s (%.2f%%)' % (desc, percent)

    return render_template('index.html', prediction=classification, image=image_path)

if __name__ == '__main__':
    app.run(port=3000, debug=True)