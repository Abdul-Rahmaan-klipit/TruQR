import cv2
import requests
import base64
from pyzbar.pyzbar import decode
import numpy as np
import time
from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity

def load_model():
    vgg16 = VGG16(weights='imagenet', include_top=False,
                pooling='avg', input_shape=(224, 224, 3))
    
    for model_layer in vgg16.layers:
        model_layer.trainable = False
    return vgg16


def load_image(image_path):
    input_image = Image.open(image_path).convert('RGB')
    resized_image = input_image.resize((224, 224))
    return resized_image


def get_image_embeddings(object_image : image):
    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = vgg16.predict(image_array)
    return image_embedding


def get_similarity_score(first_image : str, second_image : str):
    first_image = load_image(first_image)
    second_image = load_image(second_image)
    second_image = second_image.resize((224,224))
    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)
    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
    return similarity_score


def crop_inner_qr(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_edges = cv2.Canny(blurred_image, 50, 150)
    contours, _ = cv2.findContours(canny_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 1000
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area:
            x,y,w,h = cv2.boundingRect(contour)
            if h == w or h == (w + 1) or w == (h + 1) and h < 90 and w < 90:
                cropped_img = image[y:y+h, x:x+w]
                filename = f"inner_qr_code_{int(time.time())}.png"
                # cv2.imshow('Contours', cropped_img)
                cv2.imwrite(filename, cropped_img)
                return filename
    

def draw_and_save_qr(img, barcode):
  (x, y, w, h) = barcode.rect
  pts = np.array([barcode.polygon], np.int32)
  if w == h:
    is_vertical = True
  else:
    is_vertical = False

  cropped_img = 0
  if is_vertical:
    x_min, y_min = np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf

    for point in pts[0]:
        x, y = point
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    cropped_img = img[y_min:y_max, x_min:x_max]
    filename = f"qr_code_{int(time.time())}.png"
    # cv2.imshow(filename, cropped_img)
    cv2.imwrite(filename, cropped_img)

    similarity_score = get_similarity_score("resources/comparison/outer_with_inner_qr.png", filename)
    print(f"Similarity Score of Whole QR: {similarity_score}")
    if similarity_score:
        filename_1 = crop_inner_qr(cropped_img)
        if filename_1:
            similarity_score_1 = get_similarity_score("resources/comparison/border_with_inner_qr.png", filename_1)
            print(f"Similarity Score of Inner QR: {similarity_score_1}")
  return cropped_img


def capture_and_send_frame():
    color = (255, 0, 0)
    thickness = 2
    cap = cv2.VideoCapture("http://192.168.70.30:8080/video")
    # cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 70)

    while True:
        ret, frame = cap.read()
        x= 26
        frame=cv2.rectangle(frame, (400,250),(600+x,450+x),color,thickness)
        # cv2.imshow('frame', frame)
        cf=frame[250:450+x,400:600+x]
        if not ret:
            print("Failed to capture frame")
            break

        cropped_image = 0
        barcode = decode(cf)
        if barcode:
            cropped_image = draw_and_save_qr(cf.copy(), barcode[0])
            url = "http://127.0.0.1:5000/decode_qr_code"
            _, img_encoded = cv2.imencode('.jpg', cropped_image)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            response = requests.post(url , data={'image': img_base64})
            if response.status_code == 200:
                print("Response from server:", response.json())
            else:
                print("Failed to decode QR code on server")
  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

vgg16 = load_model()
capture_and_send_frame()