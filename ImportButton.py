from flask import Flask, request
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

def process_image_from_url(image_url):
    try:
        # Send a GET request to the image URL
        response = requests.get(image_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open the image using PIL (Python Imaging Library)
            image = Image.open(BytesIO(response.content))
            # Process the image here (e.g., resizing, applying filters, etc.)

            # For demonstration, let's just return the image object
            return image
        else:
            print("Failed to fetch image.")
            return None
    except Exception as e:
        print("Error processing image:", e)
        return None


@app.route('/upload-image', methods=['POST'])
def upload_image():
    data = request.json
    image_url = data.get('imageUrl')

    # Process the image URL
    processed_image = process_image_from_url(image_url)
    if processed_image:
        # Do something with the processed image (e.g., save to disk, display)
        return processed_image
    else:
        return 'Failed to process image', 500