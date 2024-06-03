import streamlit as st

from streamlit_webrtc import webrtc_streamer
import av
import cv2
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
import streamlit as st
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from sklearn.metrics.pairwise import cosine_similarity
import time
# flip = st.checkbox("Flip")
# _________________________________________________________________________




st.markdown('<h1 style="color:#5F1787; font-size:64px; font-weight:bold;">TruQR demo &copy;</h1>', unsafe_allow_html=True)
st.write("")

x1 = st.selectbox('if you are using any of the below urls kindly select the relevant.',("http://192.168.70.13:8080/video","http://192.168.70.30:8080/video",0))
st.write(f"Selected ip address: {x1}")

st.markdown('<h2 style="color:white; font-size:24px; font-weight:bold;">Webcam feed</h2>', unsafe_allow_html=True)
cap= cv2.VideoCapture(x1)
if not cap.isOpened():
    st.write(":red[Camera not found] :camera:")



frame_window= st.image([])
# DISPLAY OUTPUT
st.write("**Output**")


# yesno=False
imgfound= st.empty()
output_window = st.empty()
txt = st.empty()
st.markdown("")
txt2 = st.empty()
st.markdown("")
txt3= st.empty()
auth_disp= st.empty()


c=0


def imgfnd(yesno):
    ans=''
    with imgfound:
        if yesno==True :
            ans = "QR detected\n"
            
        #    if cropped_image.any():
            
        else:
            ans="No QR detected.\n"
        st.write(ans)
    # return yesno

imgfnd(False)

def create_box( text=""):
       text= text+ '\n'
       with txt:
        #    if yesno== True:
            st.write(text)

        #    txt.empty()
    

def create_box2( score=''):
    #    score= score 
       with txt2:
           st.write("Similarity score: \n")
           st.write(f"  {score}\n")

def create_box3( score=""):
    #    score= score 
       with txt3:
        #    if yesno== True:
        #    st.write("Similarity score: \n")
             st.write(f"  {score}\n")

def auth_box(text):
     with auth_disp:
        #    st.write("Similarity score: \n")
        # if yesno== True:
           st.write(f"  {text}\n")
#  ---------------------------------------------


        

# vgg16 = VGG16(weights='imagenet', include_top=False,   pooling='avg', input_shape=(224, 224, 3))
vgg16= ResNet50(weights="imagenet",include_top=False,pooling= 'avg', input_shape=(224, 224, 3))
def load_model():
    
    # 
    # print the summary of the model's architecture.
    vgg16.summary()
    for model_layer in vgg16.layers:
        model_layer.trainable = False
    return vgg16

import os

def load_image(image_path):
    """
        -----------------------------------------------------
        Process the image provided.
        - Resize the image
        -----------------------------------------------------
        return resized image
    """
    if os.path.exists(image_path):
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
    first_image = load_image(first_image) 
    second_image = load_image(second_image)
    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)
    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
    return similarity_score




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
    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) 
    # Perform edge detection using Canny
    canny_edges = cv2.Canny(blurred_image, 50, 150) 
    # Find contours
    contours, _ = cv2.findContours(canny_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define minimum contour area
    min_contour_area = 1000
    h,w,x,y= 0,0,0,0

    # Draw contours on the original image if contour area is greater than minimum
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area: 
            x,y,w,h = cv2.boundingRect(contour) 
    if h == w or h == (w + 1) or w == (h + 1) and h < 90 and w < 90:
        cropped_img = image[y:y+h, x:x+w]
        filename = f"inner_qr_code_{int(time.time())}.png"
        # yesno = True
        
        
        # Display the result
        # cv2.imshow('Contours', cropped_img)
        
        output_window.image(cropped_img, caption='cropped QR' )#, use_column_width=False, width=300)
        cv2.imwrite(filename, cropped_img)
        return filename
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
       
    


def draw_and_save_qr(img, barcode):
  # Get data, rectangle, and crop with padding
  imgfnd(True)
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
    cropped_img = img[y_min:y_max, x_min:x_max]

    # Filename with timestamp
    filename = f"qr_code_{int(time.time())}.png" 
    cv2.imwrite(filename, cropped_img) 
    similarity_score = get_similarity_score("outer_with_inner_qr_1.png", filename)
    
    create_box2(f"similarity score: {similarity_score}")
    
    
    if similarity_score >=0.8 :
        create_box2(f":violet[Whole QR: {similarity_score[0]*100} %]")
        create_box2(f"Whole QR: {np.round(similarity_score[0]*100,2)} % \n ")
        filename_1 = crop_inner_qr(cropped_img)
        if filename_1:
            similarity_score_1 = get_similarity_score(r"C:\Users\megha\Downloads\tiff\border_with_inner_qr_1.tiff", filename_1)
            pol= 0.5
            if similarity_score_1[0]>=pol: 
                auth_box(" :green[Product is authentic]	:white_check_mark:")
            elif similarity_score_1[0]<pol:
                create_box3("")
                auth_box(" :red[Product is not authentic]	:X:")
    else:
        auth_box(""" :red[Product is not authentic]	:X: \nIf you are not reassured, please scan again. """)
        st.markdown("")
        create_box2("")

  return cropped_img

def nein(frame):
    img= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    p=5 
    img = ski.util.img_as_ubyte(img) 
    imgr = ski.color.rgb2gray(img) 
    thresh,_= ski.filters.threshold_multiotsu(imgr) 
    print("threshold value:    ", thresh)
    bin = imgr>thresh
    cvimg = ski.util.img_as_ubyte(bin)
    return cvimg


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )



def find_squares(img): 
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt)< 1000  and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def decode_qr_code():
    response=''
    if 'image' not in request.form:
        response= 'error : No image provided, 400'

    image_base64 = request.form['image']
    image_data = base64.b64decode(image_base64)
    qr_code_info = decode(cv2.imdecode(np.frombuffer(image_data, np.uint8), -1))
    
    if qr_code_info:
        response= f'qr_code_info: {qr_code_info[0].data.decode()}), 200'
    else:
        response= 'error : No QR code detected, 400'
    create_box(response)

def capture_and_send_frame():
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2 
    cap.set(cv2.CAP_PROP_FPS, 30)
 




color = (255, 0, 0)
    # Line thickness of 2 px
thickness = 2
 

while cap.isOpened():
       ret, frame = cap.read()
       if ret: 
            x= 26
            frame=cv2.rectangle(frame, (400,250),(600+x,450+x),color,thickness)
            cf=frame[250:450+x,400:600+x]
            frame_window.image(frame)

            if not ret:
                create_box("Failed to capture frame")
                
                break

            # qr_codes = 0
            cropped_image = 0 
            qr_codes = decode(cf)
            if qr_codes: 
                cropped_image = draw_and_save_qr(cf.copy(), qr_codes[0])
                time.sleep(0.5)
                encoded_info = qr_codes[0].data.decode()
                create_box(f"Encoded information in QR code: {encoded_info}" )
                url = "http://127.0.0.1:5000/decode_qr_code"
                if cropped_image is not None:
                    _, img_encoded = cv2.imencode('.jpg', cropped_image)
                    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
cv2.destroyAllWindows()
cap.release()



st.rerun()
 