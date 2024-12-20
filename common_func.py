import re
import cv2
import numpy as np
import requests
import base64
# from google.cloud import storage
from typing import List, NamedTuple, Optional, Tuple

def generate_local_path(url):
    m = re.search('%2Fuploads%2F(.*\.jpg|\.jpeg)\?', url)
    m = re.search('\/(?:uploads|o)\/(.*(?:\.jpg|\.jpeg|\.png))\?', url)
    file_name = m.group(1)
    local_file_path = '/tmp/{}'.format(file_name)
    return file_name, local_file_path

def download_img(url, destination_name):
    response = requests.get(url)

    with open(destination_name, 'wb') as f:
        f.write(response.content)

# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)

#     blob.download_to_filename(destination_file_name)

#     print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))

def get_bounding_rect(img, low_threshold=30):
    edged = cv2.Canny(img, low_threshold, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    min_x, min_y, max_x, max_y = False, False, False, False
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        # print('debug bounding box')
        # print( (x,y,w,h) )
        if w > low_threshold and h > low_threshold:
            if min_x == False or x < min_x:
                min_x = x
            if min_y == False or y < min_y:
                min_y = y
            if max_x == False or (w+x) > max_x:
                max_x = w+x
            if max_y == False or (h+y) > max_y:
                max_y = h+y
    
    return min_x, min_y, max_x, max_y

def decode_base64_to_img(base64_img):
    decoded_data = base64.b64decode(base64_img)
    nparr = np.fromstring(decoded_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image
    
def read_image_url(image_url):
    url_response = requests.get(image_url, stream=True).raw
    content_response = requests.get(image_url).content
    
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    # return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

def resize_image(image, target_dimension):
    print( image )
    print( target_dimension )
    image_dimension = image.shape

    width = image_dimension[1]
    height = image_dimension[0]

    pad_top = 0
    pad_bottom = 0
    pad_left = 0
    pad_right = 0

    if width > height:
        scaler = width / target_dimension
        width = target_dimension
        height = height / scaler
        pad_top = int(round((width - height) / 2, 0))
        pad_bottom = int(width - height - pad_top)
    else:   
        scaler = height / target_dimension
        height = target_dimension
        width = width / scaler
        pad_left = int(round((height - width) / 2, 0))
        pad_right = int(height - width - pad_left)

    print("width & height respectively: {} & {}".format(str(width), str(height)))
    new_dimension = (int(width), int(height))
    image = cv2.resize(image, new_dimension, interpolation=cv2.INTER_AREA)

    # Pad for square image
    print("all pad top: {}, bottom: {}, left: {}, right: {}".format(str(pad_top), str(pad_bottom), str(pad_left), str(pad_right)))
    image = cv2.copyMakeBorder(image, pad_top,pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,value=[0,0,0] )
    success, encoded_image = cv2.imencode('.png', image.astype(np.uint8))
    return encoded_image.tobytes()

def encode_img_to_base64(img):
    img_encode = cv2.imencode('.png', img)[1]
    img_bytes = img_encode.tobytes()
    base64_img = base64.b64encode(img_bytes)
    return base64_img

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img

def read_image_url(url):
    print( url )
    url_response = requests.get(url, stream=True).raw
    # print( url_response )
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    # return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
