import streamlit as st
import torch
import cv2
import numpy as np
from nn_model import EnhancedCNNMoreDropout, DeepSketch
from create_models import create_maskrcnn_resnet50_fpn, create_resnet_sketch_parse_r5
from masking import image_to_mask
from scipy import ndimage
from torch.autograd import Variable
import torch
from torch import nn
from create_skeleton import create_skeleton
from skeletonization.bonetypes import common
import networkx as nx
from matplotlib import pyplot as plt
import copy
from skeletonmatching.skeleton_path import match
import matplotlib.patheffects as path_effects
from meshing.meshing import generate_mesh

WEIGHTS_PATH = 'classifier_deep_sketch_weights_200.pth'
MASK_R_CNN_PATH = 'mask_r_cnn_weights.pth'
SKETCH_PARSE_PATH = 'sketch_parse_weights.pth'
CAT_PROTOTYPE_PATH = 'cat_proto_mask.jpg'
CAT_PROTOTYPE_SEGMENTATION_PATH = 'cat_proto_segment.jpg'

joint_type_to_color = {
    common.JointType.LIMB : (0,0,255),
    common.JointType.BODY : (0,255,0),
    common.JointType.HEAD : (255,0,0),
    common.JointType.WING : (255,165,0),
    common.JointType.MIXED : (255,0,255),
    common.JointType.NONE : (0,255,255)
}

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

def cut_out_box(img, box, margin):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    img *= 255
    x_min = int(box[0])-margin
    y_min = int(box[1])-margin
    x_max = int(box[2])+margin
    y_max = int(box[3])+margin
    if(x_min < 0):
        x_min = 0
    if(y_min < 0):
        y_min = 0
    if(x_max > img.shape[1]):
        x_max = img.shape[1]
    if(y_max > img.shape[0]):
        y_max = img.shape[0]
    return img[y_min:y_max,x_min:x_max]

def cut_out_box_on_original(img, box, margin, original):
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    factor = (original.shape[1]/img.shape[1],original.shape[0]/img.shape[0])
    x_min = int((box[0]-margin)*factor[0])
    y_min = int((box[1]-margin)*factor[1])
    x_max = int((box[2]+margin)*factor[0])
    y_max = int((box[3]+margin)*factor[1])
                
    if(x_min < 0):
        x_min = 0
    if(y_min < 0):
        y_min = 0
    if(x_max > original.shape[1]):
        x_max = original.shape[1]
    if(y_max > original.shape[0]):
        y_max = original.shape[0]
    return original[y_min:y_max,x_min:x_max]


@st.cache_data
def load_model(path):
    weights = torch.load(path, map_location=torch.device('cpu'))
    model = DeepSketch()
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
    pose = np.argmax(result[4])
    output = interp(result[3]).cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)
    output = np.argmax(output, axis=2)
    return output, pose

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
    
def visualize_skeleton(mask_image,content_image , skeleton):
    resized_preprocessed = cv2.resize(content_image,(mask_image.shape[1], mask_image.shape[0]))
    skeleton_img = np.copy(mask_image)
    skeleton_img = cv2.cvtColor(skeleton_img,cv2.COLOR_GRAY2RGB)
    skeleton_img = skeleton_img * (resized_preprocessed*255)
    for bone in skeleton.get_bones():
        for point in bone.path:
            skeleton_img[int(point[1])][int(point[0])] = [255,0,0]
    for joint in skeleton.joints:
        cv2.circle(skeleton_img,(int(joint.position[0]),int(joint.position[1])),5,color=joint_type_to_color[joint.type],thickness=2)
    return skeleton_img

def get_color_map(graph):
        colors = ['magenta', 'red', 'green', 'blue', 'orange']
        color_map = []
        for _, attr in graph.nodes(data=True):
            color_map.append(colors[int(attr['type'])])
        return color_map

def draw_skeleton_graph(graph):
    fig, ax = plt.subplots()
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph,pos)
    st.pyplot(fig)

def draw_reference_and_skeleton(reference_graph, graph):
    reference_color_map = get_color_map(reference_graph)
    color_map = get_color_map(graph)
    fig, (ax1, ax2) = plt.subplots(1,2)
    pos1 = nx.get_node_attributes(reference_graph, 'pos')
    pos2 = nx.get_node_attributes(graph, 'pos')

    nx.draw(reference_graph, pos1, node_color=reference_color_map,ax=ax1)
    nx.draw(graph, pos2, node_color=color_map,ax=ax2)
    st.pyplot(fig)


st.title("🐱 or 🐶 | Sketch Classification")
input_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if input_file is not None:
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original = img.copy()
    
    st.image(img, caption='Uploaded Image', channels="BGR", use_column_width=True)

    preprocessed = preprocess(img)
    preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
    model = load_model(WEIGHTS_PATH)
    tensor = image_to_tensor(preprocessed)
    tensor = tensor.squeeze(1)
    tensor = tensor.permute(0, 3, 1, 2)

    result = predict(tensor, model)
    
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
    cropped = cut_out_box(mask_preprocessed_img, box, 25)
    cropped_from_orig = cut_out_box_on_original(mask_preprocessed_img, box, 25,original)

    st.image(image_with_box, caption="Detected Sketch", channels="RGB", use_column_width=True)
    
    st.image(cropped_from_orig, caption="Cropped from input", use_column_width=True)

    mask_image = np.transpose(mask.numpy(), (1,2,0))
    _, mask_image = cv2.threshold(mask_image, 0.5, 1, cv2.THRESH_BINARY)

    st.image(mask_image, caption="Detected Mask", use_column_width=True)

    classical_masked_image = image_to_mask(cropped_from_orig)

    longer_side_length = 331

    height, width = classical_masked_image.shape[:2]

    aspect_ratio = width / height

    if width > height:
        new_width = longer_side_length
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = longer_side_length
        new_width = int(new_height * aspect_ratio)

    classical_masked_image = cv2.resize(classical_masked_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    st.image(classical_masked_image,caption="Classical Mask", use_column_width=True)

    sketch_parse_model = load_sketch_parse_r5(SKETCH_PARSE_PATH)

    sketch_parse_preprocessed_img = preprocess_for_sketch_parse(cropped_from_orig)
 
    segment_image, estimated_pose = segment(sketch_parse_preprocessed_img, sketch_parse_model)
    st.image(colour_segmented_image(segment_image), caption="Segmented Image",use_column_width=True)
    st.write(":red[Head] :green[Body] :blue[Leg] :orange[Tail]")
    print(estimated_pose)

    pose_text = "west" if estimated_pose == 1 or estimated_pose == 2 or estimated_pose == 4 or estimated_pose == 7 else "east"
    st.write(f"Estimated subject to be looking {pose_text}")

    if(pose_text == "west"):
        segment_image = cv2.flip(segment_image, 1)
        classical_masked_image = cv2.flip(classical_masked_image,1)
        sketch_parse_preprocessed_img = cv2.flip(sketch_parse_preprocessed_img, 1)


    skeleton = create_skeleton(classical_masked_image, segment_image)
    st.image(visualize_skeleton(classical_masked_image,
                                sketch_parse_preprocessed_img,
                                skeleton),
                        caption="Image with skeleton", use_column_width=True)
    skeleton.prune_end_effectors()
    skeleton.prune_bones_without_joints()

    st.image(visualize_skeleton(classical_masked_image,
                                sketch_parse_preprocessed_img,
                                skeleton),
                        caption="Image with skeleton", use_column_width=True)
    
    prototype_img = cv2.imread(CAT_PROTOTYPE_PATH, cv2.IMREAD_GRAYSCALE)
    prototype_segmented_img = cv2.imread(CAT_PROTOTYPE_SEGMENTATION_PATH,cv2.IMREAD_GRAYSCALE)
    st.image(prototype_img, caption="Prototype mask", use_column_width=True)

    proto_skeleton = create_skeleton(prototype_img,prototype_segmented_img)

    proto_skeleton.prune_end_effectors()
    skeleton.prune_bones_without_joints()
    st.image(visualize_skeleton(prototype_img,
                                cv2.cvtColor(np.zeros_like(prototype_img), cv2.COLOR_GRAY2RGB) ,
                                proto_skeleton),
                        caption="Prototype with skeleton", use_column_width=True)
    
    
    subject_distance_transfrom = ndimage.distance_transform_edt(classical_masked_image)
    subject_normalizer_dt = 1 / (sum(sum(subject_distance_transfrom))/2)
    normalized_subject_distance_transform = subject_distance_transfrom * subject_normalizer_dt

    prototype_distance_transfrom = ndimage.distance_transform_edt(prototype_img)
    prototype_normalizer_dt = 1 / (sum(sum(prototype_distance_transfrom))/2)
    normalized_prototype_distance_transform = prototype_distance_transfrom * prototype_normalizer_dt

    fig, axs = plt.subplots(ncols=2)
    fig.suptitle("normalized distance transforms")
    axs[1].imshow(normalized_subject_distance_transform*1000, cmap="viridis")
    axs[0].imshow(normalized_prototype_distance_transform*1000, cmap="viridis")
    st.pyplot(fig)

    matched_joints = match(skeleton,~classical_masked_image,proto_skeleton,~prototype_img)

    fig, axs = plt.subplots(1,2)
    fig.suptitle("Matching between prototype and subject")
    axs[0].imshow(visualize_skeleton(prototype_img,
                            cv2.cvtColor(np.zeros_like(prototype_img), cv2.COLOR_GRAY2RGB) ,
                            proto_skeleton))
    axs[1].imshow(visualize_skeleton(classical_masked_image,
                                sketch_parse_preprocessed_img,
                                skeleton))
    axs[0].set_title("Prototype skeleton")
    axs[1].set_title("subject")
    idx = 0
    for a,b in matched_joints:
        txt1 = axs[0].text(b.position[0], b.position[1], f'J{idx+1}', color='white', fontsize=12, ha='right')
        txt2 = axs[1].text(a.position[0], a.position[1], f'J{idx+1}', color='white', fontsize=12, ha='right')
        outline_effect = [path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()]
        txt1.set_path_effects(outline_effect)
        txt2.set_path_effects(outline_effect)
        idx +=1
    st.pyplot(fig)
    
    skeleton_graph = proto_skeleton.to_network_x()
    positions = nx.get_node_attributes(skeleton_graph, 'pos') 
    for val in positions:
        print(positions[val])

    mesh_file_name = generate_mesh(classical_masked_image)
    st.write("generated_mesh")
    with open (mesh_file_name, "rb") as file:
        btn = st.download_button(
            label="Download mesh",
            data = file,
            file_name="mesh.gltf"
        )
