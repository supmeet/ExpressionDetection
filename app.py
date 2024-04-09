from flask import Flask, render_template, request, jsonify
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import warnings
import base64
import numpy as np

warnings.filterwarnings("ignore")


app = Flask(__name__, template_folder='templates')

class_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction or generation request
@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected option (prediction or generation)
    option = request.form['option']
    # option = 'prediction'

    image = request.files['image']
    # image_path = request.form['image-folder'] 

    # Perform prediction or generation based on the selected option
    selected_model = request.form['model']
    print(selected_model)
    if(selected_model == 'Model 1'):
        modelName = 'VGG+Round2.h5'
    elif(selected_model == 'Model 2'):
        modelName = 'VGGUnfreeze.keras'
    elif(selected_model == 'Model 3'):
        modelName = 'CNNmodel.keras'
    else:
        modelName = 'VGG+Round2.h5'

    if option == 'prediction':
        prediction = predict_expression(image, selected_model) 
        return render_template('prediction_result.html', image_url=get_image_url(image), original_expression="Original Expression", predicted_expression=prediction, selected_model=modelName)
    elif option == 'generation':
        generated_image,selected_model = generate_image(image, selected_model)
        return render_template('prediction_result.html', image_url=get_image_url2(generated_image), original_expression="Original Expression", predicted_expression="prediction", selected_model=selected_model)

    


def predict_expression(image, selected_model):
    # print(tf.__version__)
    if(selected_model == 'Model 1'):
        modelName = 'VGG+Round2.h5'
    elif(selected_model == 'Model 2'):
        modelName = 'VGGUnfreeze.keras'
    elif(selected_model == 'Model 3'):
        modelName = 'CNNmodel.keras'
    else:
        modelName = 'VGG+Round2.h5'

    model_path = os.path.join(os.path.dirname(__file__), modelName)
    model = load_model(model_path)
    
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    img_height, img_width = 128, 128
    if(selected_model == 'Model 1'):
        img_height, img_width = 224, 224
    image = cv2.resize(image, (img_height, img_width))
    if(selected_model == 'Model 3'):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
    else:    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0

    image = img_to_array(image)
    
    predictions = model.predict(np.expand_dims(image, axis=0))
    predicted_class_index = np.argmax(predictions)
    predicted_expression = class_labels[predicted_class_index]
    return predicted_expression

def generate_image(image,selected_model):
    if(selected_model == 'GAN'):
        modelName = "GanGenerator.h5"
    else:
        modelName = "Vae_decoder.h5"
    model_path = os.path.join(os.path.dirname(__file__), modelName)
    print(model_path)
    model = load_model(model_path)
    # print(image)
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img_height, img_width = 64, 64
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_height, img_width))
    noise = np.random.normal(0, 1, (1, 100))
    labels = np.random.randint(0, 6, 1)
    generated_images = model.predict([noise, tf.keras.utils.to_categorical(labels, 6)])
    image_bytes = tf.image.encode_jpeg(generated_images[0]).numpy()
    return image_bytes, selected_model

def get_image_url2(image_bytes):
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f'data:image/jpeg;base64,{image_base64}'
    return image_url

def get_image_url(image):
    image.seek(0)
    image_data = image.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    image_url = f'data:image/jpeg;base64,{image_base64}'
    return image_url

if __name__ == '__main__':
    app.run(debug=True)