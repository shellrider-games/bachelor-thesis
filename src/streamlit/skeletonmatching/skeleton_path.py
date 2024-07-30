from collections import defaultdict
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import networkx as nx

from skeletonmatching.graphn import optimal_sequence_bijection_rij

class SkeletonPath:
    def __init__(self, from_id, to_id, length, radii):
        self.from_id = from_id
        self.to_id = to_id
        self.length = length
        self.radii = radii

    def __str__(self):
        return "From: {0}\n To: {1}\n Length: {2}\n Radii: {3}".format(self.from_id, self.to_id, self.length, self.radii)
    

def path_distance(left: SkeletonPath, right: SkeletonPath, alpha: float):
    assert(len(left.radii) == len(right.radii))
    weighted_length = alpha * ((left.length - right.length) ** 2 / (left.length + right.length))
    distance = 0
    for i in range(len(left.radii)):
        left_radius = left.radii[i]
        right_radius = right.radii[i]
        radii_component = (left_radius - right_radius) ** 2 / (left_radius + right_radius)
        distance += radii_component
    distance += weighted_length
    return distance

def calculate_all_paths(skeleton, image, number_of_samples = 10):
    graph = skeleton.to_network_x()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    ordered_endpoints = skeleton.order_points(skeleton.find_endpoints())
    distance_transform = ndimage.distance_transform_edt(~image)
    normalizer_dt = 1 / (sum(sum(distance_transform))/2)

    paths = []
    for u in ordered_endpoints:
        for v in ordered_endpoints:
            if u == v:
                continue
            shortest_path = nx.shortest_path(graph, u, v)
            sampled_path = []
            for i in range(0,len(shortest_path)-1):
                [sampled_path.append(point) for point in skeleton.get_skeleton_path(shortest_path[i], shortest_path[i+1])]
            length = len(sampled_path)
            sample_ids = np.linspace(0, length-1, number_of_samples, dtype=int)
            sampled_radii = []
            for i in sample_ids:
                x = int(sampled_path[i][0])
                y = int(sampled_path[i][1])
                sampled_radii.append(distance_transform[y][x] * normalizer_dt)
            paths.append(SkeletonPath(u,v,length,sampled_radii))
    return paths

def match(skeleton, img, proto_skeleton, prototype_img):
    subject_paths = defaultdict(list)
    for p in calculate_all_paths(skeleton,img):
        subject_paths[p.from_id].append(p)
        
    prototype_paths = defaultdict(list)
    for p in calculate_all_paths(proto_skeleton, prototype_img):
        prototype_paths[p.from_id].append(p)

    num_subject_end_nodes = len(subject_paths)
    num_prototype_end_nodes = len(prototype_paths)

    num_dummy_nodes = max(0, num_subject_end_nodes - num_prototype_end_nodes)


    alpha = 0.01

    matrix = np.zeros((num_subject_end_nodes + 1, num_prototype_end_nodes + num_dummy_nodes + 1))
    i = 0
    for paths in subject_paths.values():
        j = 0
        for paths_prime in prototype_paths.values():
            pd_i_j = np.zeros((len(paths), len(paths_prime)))
            for k in range(len(paths)):
                for l in range(len(paths_prime)):
                    distance = path_distance(paths[k], paths_prime[l], alpha)
                    pd_i_j[k, l] = distance
            _, length = optimal_sequence_bijection_rij(pd_i_j)
            matrix[i,j] = length
            j += 1
        for d in range(num_dummy_nodes):
            matrix[i, num_prototype_end_nodes + d] = 2.0 * np.mean(matrix[:num_subject_end_nodes, :num_prototype_end_nodes])
        i+=1

    mean_value = np.mean(matrix[:-1, :-1])
    matrix[len(subject_paths), :-1] = 2.0 * mean_value
    matrix[:-1, len(prototype_paths)] = 2.0 * mean_value
    matrix[len(subject_paths), len(prototype_paths)] = 2.0 * mean_value 
    col_ind, row_ind = linear_sum_assignment(matrix)

    matching = list(zip(row_ind, col_ind))
    matched_joints = []

    for a, b in matching:
        if a >= len(subject_paths) and b >= len(prototype_paths):
            continue
        if a >= len(subject_paths):
            continue
        elif b >= len(prototype_paths):
            continue
        else:
            matched_joints.append((skeleton.get_joint(skeleton.order_points(skeleton.find_endpoints())[a]), proto_skeleton.get_joint(proto_skeleton.order_points(proto_skeleton.find_endpoints())[b])))
    return matched_joints

