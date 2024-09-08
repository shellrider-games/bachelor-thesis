from collections import defaultdict, deque
import cv2
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

def interpolate(v1, v2, val1, val2):
    if val1 != val2:
        return v1 + (v2 - v1) * (0.5 - val1) / (val2 - val1)
    else:
        return (v1 + v2) / 2

def marching_squares(image):
    rows, cols = image.shape
    vertices = []
    faces = []
    uvs = []

    lookup_table = {
        1:  [(0,0), (1,0), (0,1)],  # Bottom-left triangle
        2:  [(0,0), (1,1), (1,0)],  # Bottom-right triangle
        3:  [(0,0), (0,1), (1,0), (1,0), (0,1), (1,1)],  # Vertical edge
        4:  [(1,1), (0,1), (1,0)],  # Top-right triangle
        5:  [(0,0), (0.5,1), (0,0.5), (0,0), (0,1), (0.5,1)],  # Left vertical split
        6:  [(0,0), (1,1), (0,1), (0,0), (1,0), (1,1)],  # Horizontal edge
        7:  [(0,0), (0,1), (1,0), (0,1), (1,1), (1,0)],  # Complex top-left, bottom-right
        8:  [(0,0), (0,1), (1,1)],  # Top-left triangle
        9:  [(0,0), (1,0), (0,1), (1,0), (1,1), (0,1)],  # Diagonal edge
        10: [(0.5,0), (0.5,1), (1,0), (1,0), (0.5,1), (1,1)],  # Right vertical split
        11: [(1,0), (0,1), (1,1), (0,0), (1,0), (0,1)],  # Complex bottom-left, top-right
        12: [(0,1), (1,1), (1,0), (0,1), (1,0), (0,0)],  # Diagonal edge
        13: [(1,0), (1,1), (0,1), (0,0), (1,0), (0,1)],  # Complex bottom-right, top-left        
        14: [(0,0), (0,1), (1,0), (0,1), (1,1), (1,0)],  # Complex top-right, bottom-left
        15: [(0,0), (0,1), (1,0), (0,1), (1,1), (1,0)],  # Full squarec
    }
    
    for y in range(rows - 1):
        for x in range(cols - 1):
            square = image[y:y+2, x:x+2]
            index = square[0,0] * 1 + square[0,1]* 2 + square[1,1] * 4 + square[1,0] * 8
            if index > 0:
                base_vertices = lookup_table.get(index, [])
                face_vertices = []
                for vert in base_vertices:
                    v = (x + vert[0], y + vert[1])
                    if v not in vertices:
                        vertices.append(v)
                        uvs.append((v[0] / cols, 1 - v[1] / rows))  # Normalize UVs
                    face_vertices.append(vertices.index(v))
                for i in range(0, len(face_vertices), 3):
                    # Ensure the face is oriented clockwise
                    face = face_vertices[i:i+3]
                    if (vertices[face[1]][0] - vertices[face[0]][0]) * (vertices[face[2]][1] - vertices[face[0]][1]) - (vertices[face[1]][1] - vertices[face[0]][1]) * (vertices[face[2]][0] - vertices[face[0]][0]) > 0:
                        face = [face[0], face[2], face[1]]  # Swap last two vertices to flip
                    faces.append(face)
                    
    return np.array(vertices), np.array(faces), np.array(uvs)


def positions_match(pos1, pos2, tolerance=1e-5):
    return abs(pos1[0] - pos2[0]) < tolerance and abs(pos1[1] - pos2[1]) < tolerance

def normalize_vertices(vertices, maximum):
    vertices_normalized = []
    for vertex in vertices:
        vertices_normalized.append([vertex[0]/maximum,vertex[1]/maximum])
    
    return np.array(vertices_normalized)

def center_vertices(vertices, joints):
    center_x, center_y = np.mean(vertices, axis=0)
    vertices_centered = vertices - np.array([center_x, center_y])
    joints_centered = joints - np.array([center_x, center_y])
    return vertices_centered, joints_centered

def flip_vertically(vertices):
    vertices_flipped = vertices * np.array([1,-1])
    return vertices_flipped

def bfs_with_children(graph, start):
    visited = set()
    queue = deque([start])
    children_dict = defaultdict(list)
    
    while queue:
        node = queue.popleft()
        
        if node not in visited:
            visited.add(node)
            
            if node not in children_dict:
                children_dict[node] = []

            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    children_dict[node].append(neighbor)
                    queue.append(neighbor)
    
    return children_dict

def export_obj(vertices, faces, uvs, skinning):
    obj_lines = []

    # Write vertices
    for vertex in vertices:
        obj_lines.append(f"v {vertex[0]} {vertex[1]} 0.0")

    # Write UVs
    for uv in uvs:
        obj_lines.append(f"vt {uv[0]} {uv[1]}")

    # Write faces with vertex/texture coordinates
    for face in faces:
        obj_lines.append(f"f {face[0] + 1}/{face[0] + 1} {face[1] + 1}/{face[1] + 1} {face[2] + 1}/{face[2] + 1}")

    # Write the OBJ file to disk
    with open("export.obj", "w") as obj_file:
        obj_file.write("\n".join(obj_lines))

    with open("skinning.txt", "w") as skinning_file:
        for k, v in skinning.items():
            for vertex in v:
                skinning_file.write(f"{vertex} {k}\n")

    return "export.obj" , "skinning.txt"

def generate_mesh(image, skeleton, segmented_image, matched_joints):
    height, width = image.shape
    longer_side = height if height >= width else width
    downscale_factor = 80 / longer_side

    image = cv2.resize(image, ((int)(width * downscale_factor), (int)(height * downscale_factor)), cv2.INTER_NEAREST)
    
    # Check if segmented_image is valid
    if segmented_image is None or segmented_image.size == 0:
        raise ValueError("segmented_image is invalid or empty.")

    # Calculate new dimensions
    new_width = int(width * downscale_factor)
    new_height = int(height * downscale_factor)

    # Ensure dimensions are valid
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Calculated dimensions are invalid (<= 0). Check the downscale_factor.")

    # Perform the resize
    segmented_image = cv2.resize(segmented_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary = binary / 255

    joint_to_mapping = {}

    for a, b in matched_joints:
        joint_to_mapping[a.id] = b.id

    positions = nx.get_node_attributes(skeleton, 'pos')

    joints = []
    for idx in positions:
        joints.append(positions[idx])
    for joint in joints:
        x = joint[0]
        y = joint[1]
        joint[0] = x * downscale_factor
        joint[1] = y * downscale_factor
    joints = normalize_vertices(joints, longer_side * downscale_factor)
    vertices, faces, uvs = marching_squares(binary)
    original_vertices = vertices.copy()
    vertices = normalize_vertices(vertices, longer_side * downscale_factor)
    vertices, joints = center_vertices(vertices, joints)
    vertices = flip_vertically(vertices)
    joints = flip_vertically(joints)

    joints_dict = {}
    counter = 0
    for idx in positions:
        joints_dict[idx] = joints[counter]
        counter += 1

    # Create a KDTree for the joint positions
    joint_positions = list(joints_dict.values())
    joint_kdtree = KDTree(joint_positions)
    
    kdtree = KDTree(vertices)
    end_effectors = [node for node, degree in skeleton.degree() if degree == 1]

    vertex_classes = {i: segmented_image[int(original_vertices[i][1])][int(original_vertices[i][0])] for i in range(len(original_vertices))}
    
    # Dictionary to store nearest vertices for each end effector and their flood fill steps
    nearest_vertices = {}
    vertex_flood_fill_steps = {}  # Keeps track of the minimum flood fill steps for each vertex

    for end_effector in end_effectors:
        end_effector_pos = np.array(joints_dict[end_effector])
        
        # Find the nearest vertex to the end effector using the KDTree
        nearest_vertex_idx = kdtree.query(end_effector_pos)[1]
        
        nearest_vertex_class = vertex_classes[nearest_vertex_idx]
        
        visited = set()
        queue = deque([(nearest_vertex_idx, 0)])  # (vertex index, flood fill steps)
        vertex_list = []
        
        while queue:
            current_idx, steps = queue.popleft()
            if current_idx in visited:
                continue

            visited.add(current_idx)

            # Check if the current vertex is already assigned to another joint
            if current_idx in vertex_flood_fill_steps:
                # Reassign if this joint reaches the vertex with fewer steps
                if steps < vertex_flood_fill_steps[current_idx]:
                    nearest_vertices[current_idx] = end_effector
                    vertex_flood_fill_steps[current_idx] = steps
            else:
                # Assign this vertex to the current joint
                nearest_vertices[current_idx] = end_effector
                vertex_flood_fill_steps[current_idx] = steps

            vertex_list.append(current_idx)

            # Explore neighboring vertices from the faces connected to this vertex
            for face in faces:
                if current_idx in face:
                    for vertex_idx in face:
                        if vertex_idx not in visited and vertex_classes[vertex_idx] == nearest_vertex_class:
                            queue.append((vertex_idx, steps + 1))

    # Step 2: Ensure every vertex is assigned
    all_assigned_vertices = set(nearest_vertices.keys())
    remaining_vertices = set(range(len(vertices))) - all_assigned_vertices

    # Handle remaining vertices (like edge vertices) by assigning them to the closest joint using the joint KDTree
    for vertex_idx in remaining_vertices:
        vertex_pos = vertices[vertex_idx]
        
        # Find the nearest joint geometrically using the joint KDTree
        nearest_joint_idx = joint_kdtree.query(vertex_pos)[1]
        nearest_joint = list(joints_dict.keys())[nearest_joint_idx]
        
        # Assign the vertex to the nearest joint
        if nearest_joint is not None:
            nearest_vertices[vertex_idx] = nearest_joint

    # Now verify that every vertex has a joint assigned
    for i in range(len(vertices)):
        if i not in nearest_vertices:
            # If a vertex is still not assigned, assign it to the nearest joint geometrically
            nearest_joint_idx = joint_kdtree.query(vertices[i])[1]
            nearest_joint = list(joints_dict.keys())[nearest_joint_idx]
            nearest_vertices[i] = nearest_joint

    # Group vertices by the mapped joint
    prototype_joint_to_vertices = {}

    for vertex_idx, joint in nearest_vertices.items():
        if joint in joint_to_mapping:
            prototype_joint = joint_to_mapping[joint]
            print(f"vertices for prototype joint: {prototype_joint} : {vertex_idx}")
            if prototype_joint not in prototype_joint_to_vertices:
                prototype_joint_to_vertices[prototype_joint] = []
            prototype_joint_to_vertices[prototype_joint].append(vertex_idx)

    return export_obj(vertices, faces, uvs, prototype_joint_to_vertices)

