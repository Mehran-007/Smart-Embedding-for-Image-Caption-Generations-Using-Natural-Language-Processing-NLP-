# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from keras.utils import pad_sequences
# # Load MobileNetV2 model
# mobilenet_model = MobileNetV2(weights="imagenet")
# mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)
#
# # Load your trained model
# model = tf.keras.models.load_model('mymodel.h5')
#
# # Load the tokenizer
# with open('tokenizer.pkl', 'rb') as tokenizer_file:
#     tokenizer = pickle.load(tokenizer_file)
#
# # Process uploaded image
# def generate_caption(image_path, model, tokenizer, max_caption_length):
#     # Load image
#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#
#     # Extract features using MobileNetV2
#     image_features = mobilenet_model.predict(image, verbose=0)
#
#     # Define function to get word from index
#     def get_word_from_index(index, tokenizer):
#         return next(
#             (word for word, idx in tokenizer.word_index.items() if idx == index), None
#         )
#
#     # Generate caption using the model
#     caption = "startseq"
#     for _ in range(max_caption_length):
#         sequence = tokenizer.texts_to_sequences([caption])[0]
#         sequence = pad_sequences([sequence], maxlen=max_caption_length)
#         yhat = model.predict([image_features, sequence], verbose=0)
#         predicted_index = np.argmax(yhat)
#         predicted_word = get_word_from_index(predicted_index, tokenizer)
#         caption += " " + predicted_word
#         if predicted_word is None or predicted_word == "endseq":
#             break
#
#     # Remove startseq and endseq
#     generated_caption = caption.replace("startseq", "").replace("endseq", "")
#
#     return generated_caption
#
# # Example usage
# image_path = '390360326_26f5936189.jpg'  # Change this to the path of your image
# max_caption_length = 34
#
# generated_caption = generate_caption(image_path, model, tokenizer, max_caption_length)
# print("Generated Caption:", generated_caption)
#
# from flask import Flask, render_template, request
# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from keras.utils import pad_sequences
#
# app = Flask(__name__)
#
# # Load MobileNetV2 model
# mobilenet_model = MobileNetV2(weights="imagenet")
# mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)
#
# # Load your trained model
# model = tf.keras.models.load_model('mymodel.h5')
#
# # Load the tokenizer
# with open('tokenizer.pkl', 'rb') as tokenizer_file:
#     tokenizer = pickle.load(tokenizer_file)
#
# # Process uploaded image and generate caption
# def generate_caption(image_path, model, tokenizer, max_caption_length):
#     # Load image
#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#
#     # Extract features using MobileNetV2
#     image_features = mobilenet_model.predict(image, verbose=0)
#
#     # Define function to get word from index
#     def get_word_from_index(index, tokenizer):
#         return next(
#             (word for word, idx in tokenizer.word_index.items() if idx == index), None
#         )
#
#     # Generate caption using the model
#     caption = "startseq"
#     for _ in range(max_caption_length):
#         sequence = tokenizer.texts_to_sequences([caption])[0]
#         sequence = pad_sequences([sequence], maxlen=max_caption_length)
#         yhat = model.predict([image_features, sequence], verbose=0)
#         predicted_index = np.argmax(yhat)
#         predicted_word = get_word_from_index(predicted_index, tokenizer)
#         caption += " " + predicted_word
#         if predicted_word is None or predicted_word == "endseq":
#             break
#
#     # Remove startseq and endseq
#     generated_caption = caption.replace("startseq", "").replace("endseq", "")
#
#     return generated_caption
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/after', methods=['POST'])
# def after():
#     file = request.files['file1']
#     file.save('static/file.jpg')  # Save the uploaded image
#
#     # Generate caption for the uploaded image
#     generated_caption = generate_caption('static/file.jpg', model, tokenizer, max_caption_length=34)
#
#     return render_template('index.html', data=generated_caption)
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')


from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.utils import pad_sequences

app = Flask(__name__)

# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model('mymodel.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Process uploaded image and generate caption
def generate_caption(image_path, model, tokenizer, max_caption_length):
    # Load image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features using MobileNetV2
    image_features = mobilenet_model.predict(image, verbose=0)

    # Define function to get word from index
    def get_word_from_index(index, tokenizer):
        return next(
            (word for word, idx in tokenizer.word_index.items() if idx == index), None
        )

    # Generate caption using the model
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        caption += " " + predicted_word
        if predicted_word is None or predicted_word == "endseq":
            break

    # Remove startseq and endseq
    generated_caption = caption.replace("startseq", "").replace("endseq", "")

    return generated_caption

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['POST'])
def after():
    file = request.files['file1']
    file.save('static/file.jpg')  # Save the uploaded image

    # Generate caption for the uploaded image
    generated_caption = generate_caption('static/file.jpg', model, tokenizer, max_caption_length=34)

    return render_template('after.html', image_path='static/file.jpg', generated_caption=generated_caption)

if __name__ == '__main__':
    app.run(debug=True)
