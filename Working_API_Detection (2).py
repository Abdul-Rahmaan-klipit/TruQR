import cv2
import requests
import base64
from flask import Flask, request, jsonify
from pyzbar.pyzbar import decode
import numpy as np
import time
import skimage as ski
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image, ImageFilter
# import tensorflow

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications.vgg16 import VGG16
# from keras.applications.resnet import ResNet50
from sklearn.metrics.pairwise import cosine_similarity


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# session = requests.Session()
# retry= Retry(connect=3, backoff_factor=0.5)
# adapter= HTTPAdapter(max_retries= retry)
# session.mount('https://', adapter)


import logging
import http.client as http_client

http_client.HTTPConnection.debuglevel = 1

# logging.basicConfig() 
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True


def load_model():
    vgg16 = VGG16(weights='imagenet', include_top=False,
                pooling='avg', input_shape=(224, 224, 3))
    # vgg16= ResNet50(weights="imagenet",include_top=False,pooling= 'max', input_shape=(224, 224, 3))
    # print the summary of the model's architecture.
    vgg16.summary()
    for model_layer in vgg16.layers:
        model_layer.trainable = False
    return vgg16

def load_image(image_path):
    """
        -----------------------------------------------------
        Process the image provided.
        - Resize the image
        -----------------------------------------------------
        return resized image
    """
    input_image = Image.open(image_path).convert('RGB')
    resized_image = input_image.resize((224, 224))
    return resized_image
def get_image_embeddings(object_image : image):
    """
      -----------------------------------------------------
      convert image into 3d array and add additional dimension for model input
      -----------------------------------------------------
      return embeddings of the given image
    """
    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = vgg16.predict(image_array)
    return image_embedding
def get_similarity_score(first_image : str, second_image : str):
    """
        -----------------------------------------------------
        Takes image array and computes its embedding using VGG16 model.
        -----------------------------------------------------
        return embedding of the image
    """
    # vgg16 = load_model()
    first_image = load_image(first_image)
    
    second_image = load_image(second_image)
    second_image = second_image.resize((224,224))
    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)
    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
    return similarity_score
# def show_image(image_path):
#   image = mpimg.imread(image_path)
#   imgplot = plt.imshow(image)
#   plt.show()
# vgg16 = load_model()
# sunflower = r'C:\Users\abdul\Downloads\border_with_inner_qr_1.png'
# helianthus = r"C:\Users\abdul\Downloads\contour_1.png"
# similarity_score = get_similarity_score(sunflower, helianthus)
# print(similarity_score)

def scharr(image, inp):
    scharrx=0
    scharry=0
    if inp =='sc':
        scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)  # X direction
        scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)  # Y direction
    if inp== "sob":
        scharrx = cv2.Sobel(image, cv2.CV_64F, 1, 0)  # X direction
        scharry = cv2.Sobel(image, cv2.CV_64F, 0, 1)  # Y direction
    # Convert back to uint8
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.convertScaleAbs(scharry)
    # Combine the two gradients
    scharr_combined = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    # Display the result
    return scharr_combined
def preproc(frame, inp):
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame= scharr(frame, inp)
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    blurred= cv2.threshold(blurred,200, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)[1]
    img= Image.fromarray(blurred)
    img= img.filter(ImageFilter.FIND_EDGES)
    img = np.array(img)

def crop_inner_qr(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray image',gray_image)
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # cv2.imshow('blurred_image', blurred_image)
    # Perform edge detection using Canny
    canny_edges = cv2.Canny(blurred_image, 50, 150)
    # cv2.imshow('canny image',canny_edges)
    # Find contours
    contours, _ = cv2.findContours(canny_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define minimum contour area
    min_contour_area = 1000

    # Draw contours on the original image if contour area is greater than minimum
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area:
            # cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            x,y,w,h = cv2.boundingRect(contour)
            # cv2.putText(image, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
    if h == w or h == (w + 1) or w == (h + 1) and h < 90 and w < 90:
        cropped_img = image[y:y+h, x:x+w]
        filename = f"inner_qr_code_{int(time.time())}.png"
        # Display the result
        cv2.imshow('Contours', cropped_img)
        cv2.imwrite(filename, cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        return filename
    

def draw_and_save_qr(img, barcode):
  # Get data, rectangle, and crop with padding
  (x, y, w, h) = barcode.rect
  # cropped_img = img[y-10:y+h+10, x-10:x+w+10]
  # Get polygon points
  pts = np.array([barcode.polygon], np.int32)
  # print(pts)
  # Check if QR code is vertical (optional)
  # Calculate width and height based on polygon points
  #width = abs(pts[0][0][0] - pts[0][2][0])
  #height = abs(pts[0][0][1] - pts[0][1][1])

  # Check if height is greater than width (tolerance for minor variations)
  if w == h:
    is_vertical = True
  else:
    is_vertical = False

  #is_vertical = height > (width * 0.1)  # Adjust tolerance as needed
  cropped_img = 0
  if is_vertical:
    # Code is vertical, proceed with cropping and saving
    # ... (rest of the code for calculating min/max, cropping, saving)

    # Initialize variables to store min and max values
    x_min, y_min = np.inf, np.inf  # Start with infinity for minimums
    x_max, y_max = -np.inf, -np.inf  # Start with negative infinity for maximums

    # Iterate through each point in the single element list of pts
    for point in pts[0]:
        x, y = point
        x_min = min(x_min, x)  # Update min x if current x is smaller
        y_min = min(y_min, y)  # Update min y if current y is smaller
        x_max = max(x_max, x)  # Update max x if current x is larger
        y_max = max(y_max, y)  # Update max y if current y is larger
    #
    # Crop the frame based on polygon points
    cropped_img = img[y_min-5:y_max+5, x_min-5:x_max+5]
    # Filename with timestamp
    filename = f"qr_code_{int(time.time())}.png"
    cv2.imshow(filename, cropped_img)
    cv2.imwrite(filename, cropped_img)
#   # Draw on original image
#   pts = np.array([barcode.polygon], np.int32).reshape((-1, 1, 2))
#   cv2.polylines(img, [pts], True, (255, 0, 255), 5)
#   cv2.putText(img, data, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

#   print(f"QR data: {data}, Saved as: {filename}")
    similarity_score = get_similarity_score("outer_with_inner_qr_1.png", filename)
    print(f"Similarity Score of Whole QR: {similarity_score}")
    if similarity_score:
        filename_1 = crop_inner_qr(cropped_img)
        if filename_1:
            similarity_score_1 = get_similarity_score("border_with_inner_qr_1_1.png", filename_1)
            print(f"Similarity Score of Inner QR: {similarity_score_1}")
  return cropped_img

def nein(frame):
    img= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    p=5
    # img =  cv.GaussianBlur(img,(p,p),0)
    # img= cv.medianBlur(img,5)
    img = ski.util.img_as_ubyte(img)
    # img= ski.exposure.equalize_adapthist(img, kernel_size=None, clip_limit=0.002, nbins=256)
    imgr = ski.color.rgb2gray(img)
    # thresh = ski.filters.threshold_otsu(imgr)
    thresh,_= ski.filters.threshold_multiotsu(imgr)
    # thresh = ski.filters.threshold_li(imgr)
    # thresh = ski.filters.threshold_isodata(imgr)
    # thresh = ski.filters.threshold_minimum(imgr)
    # thresh = ski.filters.threshold_triangle(imgr)   #inverts img, useless
    # thresh = ski.filters.threshold_yen(imgr)  #requires good lighting
    # thresh = ski.filters.threshold_mean(imgr)
    print("threshold value:    ", thresh)
    bin = imgr>thresh
    cvimg = ski.util.img_as_ubyte(bin)
    return cvimg

app = Flask(__name__)


import asyncio
import random

@app.route('/decode_qr_code', methods=['POST'])
def decode_qr_code():
    print('enteredddd')
    if 'image' not in request.form:
        return jsonify({'error': 'No image provided'}), 400

    image_base64 = request.form['image']
    image_data = base64.b64decode(image_base64)
    cf= cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    barcode = decode(cf)
    cropped_image = draw_and_save_qr(cf.copy(), barcode[0])
    # qr_code_info = decode(cf)

    if barcode:
        return jsonify({'qr_code_info': barcode[0].data.decode()}), 200
    else:
        return jsonify({'error': 'No QR code detected'}), 400
    # await asyncio.sleep(100/1000)


# from requests.adapters import HTTPAdapter

def capture_and_send_frame():
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    # Display the resulting frame


    cap = cv2.VideoCapture("http://192.168.70.13:8080/video")
    # cap = cv2.VideoCapture(0)
 
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
         
        x= 26
        frame=cv2.rectangle(frame, (400,250),(600+x,450+x),color,thickness)
        cv2.imshow('frame', frame)
        cf=frame[250:450+x,400:600+x]
        if not ret:
            print("Failed to capture frame")
            break
 
        cropped_image = 0
         
        if ret:
            # cropped_image = draw_and_save_qr(cf.copy(), barcode[0])
            # print(qr_codes)
            # encoded_info = qr_codes[0].data.decode()
            # print("Encoded information in QR code:", encoded_info)
            # t1= time.time()
            # print(t1)
            url = "http://127.0.0.1:5000/decode_qr_code"
            _, img_encoded = cv2.imencode('.jpg', cf)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8') 


            if img_base64:
                 
                response = requests.post(url, data={'image': img_base64} , verify=False) 

                if response.status_code == 200:
                    print("Response from server:", response.json()) 
                else:
                    print("Failed to decode QR code on server") 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # from threading import Thread
    vgg16 = load_model()
    # capture_thread = Thread(target=capture_and_send_frame)
    # capture_thread.start()
    capture_and_send_frame()
    app.run(debug=True, host='0.0.0.0')
