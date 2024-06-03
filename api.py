from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from pyzbar.pyzbar import decode

app = Flask(__name__)

@app.route('/decode_qr_code', methods=['POST'])
def decode_qr_code():
    if 'image' not in request.form:
        return jsonify({'error': 'No image provided'}), 400

    image_base64 = request.form['image']
    image_data = base64.b64decode(image_base64)
    qr_code_info = decode(cv2.imdecode(np.frombuffer(image_data, np.uint8), -1))

    if qr_code_info:
        return jsonify({'qr_code_info': qr_code_info[0].data.decode()}), 200
    else:
        return jsonify({'error': 'No QR code detected'}), 400

if __name__ == '__main__':
    app.run(debug=True)