from skeletonization.skeletonizer import Skeletonizer
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from skeletonization.bonetypes import common
import networkx as nx

def create_skeleton(img, segmented_img):
    img = ~img
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    skeletonizer = Skeletonizer(img)
    skeleton = skeletonizer.skeletonize(30,3,10)
    segmented_img = segmented_img.astype(np.uint8)
    segmented_img = cv2.resize(segmented_img,(img.shape[1],img.shape[0]))
    skeleton.identify(segmented_img)
    return skeleton

def create_quadruped_graph():
    """
    @brief Creates a quadruped skeleton.
    @returns Skeleton type, X
    """

    quad_like = nx.Graph()
    quad_like.add_node(0, pos=np.asarray([0.25, 0], dtype=float), type=common.JointType.LIMB)
    quad_like.add_node(1, pos=np.asarray([0.50, 0], dtype=float), type=common.JointType.LIMB)
    quad_like.add_node(2, pos=np.asarray([0.62, 0], dtype=float), type=common.JointType.LIMB)
    quad_like.add_node(3, pos=np.asarray([0.93, 0], dtype=float), type=common.JointType.LIMB)
    quad_like.add_node(4, pos=np.asarray([0.37, 1], dtype=float), type=common.JointType.BODY)
    quad_like.add_node(5, pos=np.asarray([0.75, 1], dtype=float), type=common.JointType.BODY)
    quad_like.add_node(6, pos=np.asarray([1, 0.80], dtype=float), type=common.JointType.BODY)
    quad_like.add_node(7, pos=np.asarray([0, 1], dtype=float), type=common.JointType.HEAD)

    quad_like.add_edge(0, 4, type=common.BoneType.MIXED)
    quad_like.add_edge(1, 4, type=common.BoneType.MIXED)
    quad_like.add_edge(2, 5, type=common.BoneType.MIXED)
    quad_like.add_edge(3, 5, type=common.BoneType.MIXED)
    quad_like.add_edge(4, 5, type=common.BoneType.BODY)
    quad_like.add_edge(5, 6, type=common.BoneType.BODY)
    quad_like.add_edge(4, 7, type=common.BoneType.MIXED)

    return quad_like