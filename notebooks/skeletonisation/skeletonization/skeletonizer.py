import numpy as np
from cv2 import findContours, fillPoly, CHAIN_APPROX_NONE, RETR_EXTERNAL
from skimage import color, filters
import networkx as nx

from skeletonization.bonetypes.common import JointType, BoneType
from skeletonization.helper.ussr import USSR
from skeletonization.helper.voronoi_skeleton import skeletonize
from matplotlib import pyplot as plt

class Bone:
    """
    @brief Class representing a Bone.
    """

    def __init__(self, id: int, type: BoneType, id_j1: int, id_j2: int):
        """
        @brief Constructor
        @param id ID of the bone
        @param type Type of the bone
        @param id_j1 ID of the first joint
        @param id_j2 ID of the second joint
        """
        self.id = id
        self.type = type
        self.attached_joints = [id_j1, id_j2]

    def __str__(self):
        return "bone({}, {}, {} -> {})".format(self.id, self.type, self.attached_joints[0], self.attached_joints[1])

    def is_made(self, id1: int, id2: int) -> bool:
        return id1 in self.attached_joints and id2 in self.attached_joints


class Joint:
    """
    @brief Class representing a Joint.
    """

    def __init__(self, id: int, type: JointType, position: np.array):
        """
        @brief Constructor
        @param id ID of the joint
        @param type Type of the Joint
        @param position Position of the joint
        """
        self.id = id
        self.type = type
        self.position = position


class Skeleton:
    """
    @brief Class representing a Skeleton consisting of bones and joints.
    """

    def __init__(self):
        """
        @brief Default Constructor
        """
        self.bones = []
        self.skeleton_path = []
        self.joints = []
        pass

    def __joint_index(self, position: np.array, threshold: int) -> int:
        """
        @brief Returns the index of the joint defined by the position.
        @param position Position of the joint
        @param threshold Threshold for distance between unique joints
        @return ID of the joint or -1 if it does not exist
        """
        for i in range(len(self.joints)):
            if np.linalg.norm(np.asarray(self.joints[i].position) - np.asarray(position)) <= threshold:
                return i
        return -1

    def __bone_index(self, id_j1: int, id_j2: int) -> int:
        """
        @brief Returns the index of the bone defined by the two given
        joint ids.
        @param id_j1 Joint ID A
        @param id_j2 Joint ID 2
        @return ID of the bone or -1 if it does not exist
        """
        for i in range(len(self.bones)):
            if (self.bones[i].attached_joints[0] == id_j1 and self.bones[i].attached_joints[1] == id_j2) \
                    or (self.bones[i].attached_joints[0] == id_j2 and self.bones[i].attached_joints[1] == id_j1):
                return i
        return -1

    def add_joint(self, position: np.array, threshold: int = 20) -> int:
        """
        @brief Adds a new joint to the skeleton or returns the id of an existing joint
        at the given position.
        @param position Position of the joint
        @param threshold Threshold for joints to be unique
        @return ID of the added joint (or the existing one)
        """
        j_id = self.__joint_index(position, threshold)
        if j_id != -1:
            return j_id
        j = Joint(len(self.joints), JointType.NONE, position)
        self.joints.append(j)
        return j.id

    def add_bone(self, id_j1: int, id_j2: int, path=None) -> int:
        """
        @brief Adds a new bone to the skeleton based on joint ids if the bone does not
        exist, otherwise it just returns the bone id.
        @param id_j1 Id of the first joint
        @param id_j2 Id of the second joint
        @param path Sampled skeleton path along the bone (edge)
        @return ID of the added bone (or the existing one)
        """
        b_id = self.__bone_index(id_j1, id_j2)
        if b_id != -1:
            return b_id
        b = Bone(len(self.bones), BoneType.NONE, id_j1, id_j2)
        self.bones.append(b)
        self.skeleton_path.append(path)
        return b.id

    def identify(self, segmented_image):
        """
        @brief Identify the joints and bones based on the segmented image
        @param segmented_image Segmented Image
        @return None
        """
        for joint in self.joints:
            joint.type = JointType(segmented_image[round(joint.position[1]), round(joint.position[0])])

        for bone in self.bones:
            if self.joints[bone.attached_joints[0]].type == self.joints[bone.attached_joints[1]].type:
                bone.type = BoneType(int(self.joints[bone.attached_joints[0]].type))
            else:
                bone.type = BoneType.MIXED

    def normalize_positions(self, toInt: bool = False):
        """
        @brief Normalizes the joint positions to [0, 1] interval
        @param toInt If true, it will approximate the interval [0, 1] to [0, 100] interval
        """
        valX = np.asarray([1000, -1000])
        valY = valX
        for joint in self.joints:
            valX[1] = max(valX[1], joint.position[0])
            valX[0] = min(valX[0], joint.position[0])
            valY[1] = max(valY[1], joint.position[1])
            valY[0] = min(valY[0], joint.position[1])

        xRatio = valX[1] - valX[0]
        yRatio = valY[1] - valY[0]
        for joint in self.joints:
            joint.position[0] = (joint.position[0] - valX[0]) / xRatio
            joint.position[1] = (joint.position[1] - valY[0]) / yRatio
            if toInt:
                joint.position[0] *= 100
                joint.position[1] *= 100

    def normalize_and_flip_positions(self, toInt:bool = False):
        self.normalize_positions(toInt)
        for joint in self.joints:
            joint.position[1] = 1 - joint.position[1]

    def to_network_x(self) -> nx.Graph:
        """
        @brief Creates an networkX graph from the skeleton
        @return Graph
        """
        graph = nx.Graph()
        for joint in self.joints:
            graph.add_node(joint.id, pos=np.asarray(joint.position), type=joint.type)
        for bone in self.bones:
            graph.add_edge(bone.attached_joints[0], bone.attached_joints[1], type=bone.type)

        return graph

    def get_bones(self):
        return list(zip(self.bones, self.skeleton_path))

    def get_joint(self, id: int):
        return self.joints[id]

    def get_skeleton_path(self, u: int, v: int):
        for bone, skeleton in self.get_bones():
            if bone.is_made(u, v):
                return skeleton
        raise Exception("no bone with these ids ({}, {}) exists".format(u, v))
    
    def find_points_to_remove(self) -> list:
        """
        @brief Find all non endpoints in the skeleton
        @return List of non enpoint joint IDs
        """
        ids = []
        for joint in self.joints:
            connected_bones = 0
            for bone in self.bones:
                if joint.id in bone.attached_joints:
                    connected_bones += 1
            if connected_bones != 1:
                ids.append(joint.id)
        return ids
    
    def find_endpoints(self) -> list:
        """
        @brief Find endpoints in the skeleton
        @return List of endpoint joint IDs
        """
        endpoint_ids = []
        for joint in self.joints:
            connected_bones = 0
            for bone in self.bones:
                if joint.id in bone.attached_joints:
                    connected_bones += 1
            if connected_bones == 1:
                endpoint_ids.append(joint.id)
        return endpoint_ids
    
    def order_points(self, endpoints) -> list:
        """
        Order the endpoints along the contour in a clockwise direction.
        """
        center = np.mean([self.joints[ep].position for ep in endpoints], axis=0)
        ordered_endpoints = sorted(endpoints, key=lambda ep: np.arctan2(self.joints[ep].position[1] - center[1], self.joints[ep].position[0] - center[0]))
        return ordered_endpoints


class Skeletonizer:
    """
    @brief Class for skeletonizing sketches.
    """

    def __init__(self, image):
        """
        @brief Constructor Prepares the given image for skeletonization.
        """
        self.joint_types = []
        self.image = color.rgb2gray(image)
        self.image = np.invert(self.image > filters.threshold_otsu(self.image))
        self.shape = np.zeros(self.image.shape)
        self.ussr = None

    def skeletonize(self, threshold: int, residual_type: int, merge_threshold: int = 20) -> Skeleton:
        """
        @brief Creates a skeleton for the given image.
        @param threshold Defines the threshold for the voronoi skeletonization
        @param residual_type Defines the residual type for voronoi skeletonization
        @param merge_threshold Threshold for unique joints (implying unique bones)
        @returns Skeleton
        """
        skeleton = Skeleton()

        contours = findContours(self.image.astype(np.uint8), RETR_EXTERNAL, CHAIN_APPROX_NONE)[-2]
        fillPoly(self.shape, pts=contours, color=(255, 255, 255))
        self.shape = self.shape.astype(bool)

        vrn_skeleton = skeletonize(self.shape, threshold, residual_type)
        self.ussr = USSR.from_shape(self.shape, vrn_skeleton)
        self.ussr.flip()

        poly = self.ussr.polygon
        special_points = self.ussr.get_special_points()

        line = []


        for poly_point in poly:
            line.append([poly_point[1], poly_point[0]])
            for special_point in special_points:
                if poly_point[0] == poly[special_point, 0] and poly_point[1] == poly[special_point, 1]:
                    if len(line) > 2:
                        skeleton.add_bone(skeleton.add_joint(line[0], merge_threshold),
                                          skeleton.add_joint(line[len(line) - 1], merge_threshold), line.copy())
                        line.clear()

        #plt.show()
        return skeleton

    def get_polygon(self):
        """
        @brief Returns the list of points
        @return Lists of polygon points
        """
        return self.ussr.polygon

    def is_end_point(self, point):
        """
        @brief Checks if the point is an end point.
        @param point Point to be checked using the ussr polygon.
        @returns True if it is an end point.
        """
        end_points = self.ussr.get_end_points()
        poly = self.ussr.polygon
        for p in end_points:
            if (poly[p, 0] == point[0] and poly[p, 1] == point[1]) or (
                    poly[p, 1] == point[0] and poly[p, 0] == point[1]):
                return True
        return False
