from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html',t=0)

model = tf.keras.models.load_model('D:/Coding/2.Mine Project/MCA Multi Image Classification/Multi Image Classification Model.h5')
print(model)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    print(type(file),"this is a file")

    if file.filename == '':
        return 'No selected file'

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = image.load_img(file_path, target_size=(150, 150))  
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        class_labels = ['buildings','forest','glacier','mountain','sea','street']

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]

        print(f'The predicted class is: {predicted_class} - {predicted_label}')

        return render_template('index.html', filename=file_path, predicted_class=predicted_class, predicted_label=predicted_label,t=1)





@app.route('/uploads/<filename>')
def display_image(filename):
    print("hi krishna",filename)
    return redirect(url_for('../uploads',filename=filename))

if __name__ == '__main__':
    app.run(debug=True)
