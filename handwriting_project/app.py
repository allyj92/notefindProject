from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from flask_cors import CORS  # Import CORS to handle cross-origin requests

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# EMNIST byclass dataset includes 62 classes (digits 0-9, uppercase A-Z, lowercase a-z)
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits 0-9
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Uppercase A-Z
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'  # Lowercase a-z
]

# Load the pre-trained model
try:
    model = load_model('emnist_cnn_model.keras')
    print("Model loaded successfully.")
    model.summary()  # Print the model summary to verify it's loaded correctly
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Function to preprocess the image received from the client
def preprocess_image(image_data):
    try:
        # Convert the base64-encoded image data back to an image
        image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))

        # Resize the image to 28x28 pixels and convert it to grayscale (EMNIST input size)
        image = image.resize((28, 28)).convert('L')

        # Convert the image to a numpy array, reshape for the model, and normalize pixel values (0-1)
        image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0

        # Debugging logs
        print(f"Image array shape: {image_array.shape}")
        print(f"Image array (first 10 pixels): {image_array.flatten()[:10]}")

        return image_array
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request sent by the client
        image_data = request.json.get('image')
        print("Received image data from client.")
        
        # Preprocess the image to make it ready for model prediction
        image = preprocess_image(image_data)

        if image is None:
            return jsonify({'error': 'Image processing failed'}), 400
        
        print(f"Preprocessed image shape: {image.shape}")
        
        # Get the prediction probabilities from the model
        prediction = model.predict(image)
        print(f"Prediction Probabilities: {prediction}")  # Log the prediction probabilities
        
        # Find the index of the class with the highest probability
        predicted_class_index = np.argmax(prediction)
        
        # Map the predicted class index to the corresponding character (digit/letter)
        predicted_character = emnist_labels[predicted_class_index]
        print(f"Predicted Character: {predicted_character}")
        
        # Return the prediction as a JSON response back to the client
        return jsonify({'prediction': predicted_character})
    
    except Exception as e:
        # Handle any errors that occur during prediction
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)