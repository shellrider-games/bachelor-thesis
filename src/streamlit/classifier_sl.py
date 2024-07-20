import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from nn_model import EnhancedCNNMoreDropout
from create_models import create_maskrcnn_resnet50_fpn, create_resnet_sketch_parse_r5
from masking import image_to_mask
from scipy import ndimage
from torch.autograd import Variable
import torch
from torch import nn, device

WEIGHTS_PATH = 'weights.pth'
MASK_R_CNN_PATH = 'mask_r_cnn_weights.pth'
SKETCH_PARSE_PATH = 'sketch_parse_weights.pth'

def preprocess(image, size = (128, 128)):
    preprocessed = cv2.resize(image, size)
    preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    _, preprocessed = cv2.threshold(preprocessed, 127, 1, cv2.THRESH_BINARY)
    preprocessed = preprocessed.astype(np.float32)
    return preprocessed
    
def preprocess_for_sketch_parse(image):
    preprocessed = cv2.resize(image,(321,321),interpolation=cv2.INTER_CUBIC)
    preprocessed = ndimage.grey_erosion(preprocessed[:,:,0].astype(np.uint8), size=(2,2))
    preprocessed = np.repeat(preprocessed[:,:,np.newaxis],3,2)
    return preprocessed

def image_to_tensor(image):
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def draw_box_on_image(img, box):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    img *= 255
    x_min = int(box[0])
    y_min = int(box[1])
    x_max = int(box[2])
    y_max = int(box[3])

    img[y_min:y_max+1, x_min] = [255, 0, 0]
    img[y_min:y_max+1, x_max] = [255, 0, 0]
    img[y_min, x_min:x_max+1] = [255, 0, 0]
    img[y_max, x_min:x_max+1] = [255, 0, 0]
    return img

def cut_out_box(img, box):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    img *= 255
    x_min = int(box[0])-10
    y_min = int(box[1])-10
    x_max = int(box[2])+10
    y_max = int(box[3])+10
    if(x_min < 0):
        x_min = 0
    if(y_min < 0):
        y_min = 0
    if(x_max > img.shape[1]):
        x_max = img.shape[1]
    if(y_max > img.shape[0]):
        y_max = img.shape[0]
    return img[y_min:y_max,x_min:x_max]


@st.cache_data
def load_model(path):
    weights = torch.load(path, map_location=torch.device('cpu'))
    model = EnhancedCNNMoreDropout()
    model.load_state_dict(weights)
    model.eval()
    return model

@st.cache_data
def load_mask_r_cnn(path):
    weights = torch.load(path, map_location=torch.device('cpu'))
    model = create_maskrcnn_resnet50_fpn()
    model.load_state_dict(weights['state_dict'])
    model.eval()
    return model

@st.cache_data
def load_sketch_parse_r5(path):
    weights = torch.load(path, map_location=torch.device('cpu'))
    model = create_resnet_sketch_parse_r5()
    model.load_state_dict(weights)
    model.eval()
    return model

def predict(image, model):
    model.eval()
    with torch.inference_mode():
        result = model(image)
    return result

def segment(image, model):
    model.eval()
    with torch.no_grad():
        result = model([Variable(torch.from_numpy(image[np.newaxis,:].transpose(0,3,1,2)).float()),0])
    interp = nn.UpsamplingBilinear2d(size=(321,321))
    output = interp(result[3]).cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    output = np.argmax(output, axis=2)
    return output

def colour_segmented_image(image):
    colors = {
        0: [0, 0, 0],       # Nothing
        1: [255, 0, 0],     # Head
        2: [0, 255, 0],     # Body
        3: [0, 0, 255],     # Leg
        4: [255, 165, 0]    # Tail
    }
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for value, color in colors.items():
        rgb_image[image == value] = color
    return rgb_image
    

st.title("🐱 or 🐶 | Sketch Classification")
input_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if input_file is not None:
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(img, caption='Uploaded Image', channels="BGR", use_column_width=True)
    
    preprocessed = preprocess(img)
    model = load_model(WEIGHTS_PATH)
    result = predict(image_to_tensor(preprocessed), model)
    
    result = result.squeeze().item()
    emoji = "🐱" if result >= 0.5 else "🐶"
    
    st.header(f"I believe this sketch contains a {emoji}")

    mask_r_cnn_model = load_mask_r_cnn(MASK_R_CNN_PATH)
    mask_preprocessed_img = preprocess(img, (331, 331))
    st.image(mask_preprocessed_img, caption='Preprocessed for MaskRCNN', use_column_width=True)

    img_tensor = image_to_tensor(mask_preprocessed_img)
    result = predict(img_tensor, mask_r_cnn_model)
    box = result[0]["boxes"][0]
    mask = result[0]["masks"][0]

    image_with_box = draw_box_on_image(mask_preprocessed_img, box)
    cropped = cut_out_box(mask_preprocessed_img, box)

    st.image(image_with_box, caption="Detected Sketch", channels="RGB", use_column_width=True)

    mask_image = np.transpose(mask.numpy(), (1,2,0))
    _, mask_image = cv2.threshold(mask_image, 0.5, 1, cv2.THRESH_BINARY)

    st.image(mask_image, caption="Detected Mask", use_column_width=True)

    classical_masked_image = image_to_mask(cropped)
    st.image(classical_masked_image,caption="Classical Mask", use_column_width=True)

    sketch_parse_model = load_sketch_parse_r5(SKETCH_PARSE_PATH)

    sketch_parse_preprocessed_img = preprocess_for_sketch_parse(cropped)
    st.image(sketch_parse_preprocessed_img, caption="Preprocessed for segmentation", use_column_width=True)

    segment_image = segment(sketch_parse_preprocessed_img, sketch_parse_model)
    st.image(colour_segmented_image(segment_image), caption="Segmented Image",use_column_width=True)
    st.write(":red[Head] :green[Body] :blue[Leg] :orange[Tail]")