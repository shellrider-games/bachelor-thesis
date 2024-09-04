from collections import defaultdict, deque
import cv2
import numpy as np
import pygltflib
import base64
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
        4:  [(1,1), (0,1), (1,0)],  # Top-right triangle
        8:  [(0,0), (0,1), (1,1)],  # Top-left triangle
        3:  [(0,0), (0,1), (1,0), (1,0), (0,1), (1,1)],  # Vertical edge
        6:  [(0,0), (1,1), (0,1), (0,0), (1,0), (1,1)],  # Horizontal edge
        9:  [(0,0), (1,0), (0,1), (1,0), (1,1), (0,1)],  # Diagonal edge
        12: [(0,1), (1,1), (1,0), (0,1), (1,0), (0,0)],  # Diagonal edge
        5:  [(0,0), (0.5,1), (0,0.5), (0,0), (0,1), (0.5,1)],  # Left vertical split
        10: [(0.5,0), (0.5,1), (1,0), (1,0), (0.5,1), (1,1)],  # Right vertical split
        7:  [(0,0), (0,1), (1,0), (0,1), (1,1), (1,0)],  # Complex top-left, bottom-right
        14: [(0,0), (0,1), (1,0), (0,1), (1,1), (1,0)],  # Complex top-right, bottom-left
        13: [(1,0), (1,1), (0,1), (0,0), (1,0), (0,1)],  # Complex bottom-right, top-left
        11: [(1,0), (0,1), (1,1), (0,0), (1,0), (0,1)],  # Complex bottom-left, top-right
        15: [(0,0), (0,1), (1,0), (0,1), (1,1), (1,0)],  # Full square
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

def export_gltf(vertices, faces, joints, skeleton_graph):
    points = np.zeros((vertices.shape[0], 3),dtype=np.float32)
    points[:,:2] = vertices.astype(np.float32)

    triangles = faces.astype(np.uint32)
    
    triangles_binary_blob = triangles.flatten().tobytes()
    points_binary_blob = points.tobytes()

    closeness_centrality = nx.closeness_centrality(skeleton_graph)
    root_start_idx = max(closeness_centrality, key=closeness_centrality.get)
    discovery_order = bfs_with_children(skeleton_graph, root_start_idx)

    bones = []

    node_index_map = {}

    for i, parent in enumerate(discovery_order.keys()):
        x, y = joints[parent]
        translation = [x,y,0.0]

        gltf_node = pygltflib.Node(name=f"Bone_{parent}", translation=translation)
        bones.append(gltf_node)
        node_index_map[parent] = i


    for parent, children in discovery_order.items():
        parent_index = node_index_map[parent]
        parent_coords = joints[parent]

        for child in children:
            child_index = node_index_map[child]
            child_coords = joints[child]

            relative_translation = [
                child_coords[0] - parent_coords[0],
                child_coords[1] - parent_coords[1],
                0.0
            ]

            bones[child_index].translation = relative_translation

        bones[parent_index].children = [node_index_map[child] for child in children]

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0,len(bones)])],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=1), indices=0
                    )
                ]
            )
        ],
        accessors=[
            # Accessor for triangle indices
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_INT,
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
            # Accessor for vertex positions
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(triangles_binary_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(triangles_binary_blob),
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=len(triangles_binary_blob) + len(points_binary_blob)
            )
        ],
    )

    # Add all nodes to the GLTF model
    gltf.nodes.extend(bones)

    gltf.nodes.append(
        pygltflib.Node(
            name="Mesh",
            mesh=0
        )
    )

    binary_blob = triangles_binary_blob + points_binary_blob
    base64_blob = base64.b64encode(binary_blob).decode('utf-8')
    gltf.buffers[0].uri = f"data:application/octet-stream;base64,{base64_blob}"
    gltf.save("export.gltf")
    return "export.gltf"

def generate_mesh(image, skeleton ,segmented_image, matched_joints):
    height, width = image.shape
    longer_side = height if height >= width else width
    downscale_factor = 80/longer_side

    image = cv2.resize(image, ((int)(width*downscale_factor),(int)(height*downscale_factor)), cv2.INTER_NEAREST)
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
    binary = binary/255

    joint_to_mapping = {}

    for a,b in matched_joints:
        joint_to_mapping[a.id] = b.id

    positions = nx.get_node_attributes(skeleton, 'pos')

    joints = []
    for idx in positions:
        joints.append(positions[idx])
    for joint in joints:
        x = joint[0]
        y = joint[1]
        joint[0] = x*downscale_factor
        joint[1] = y*downscale_factor
    joints = normalize_vertices(joints,longer_side*downscale_factor)
    vertices, faces, uvs = marching_squares(binary)
    original_vertices = vertices.copy()
    vertices = normalize_vertices(vertices,longer_side*downscale_factor)
    vertices, joints = center_vertices(vertices, joints)
    vertices = flip_vertically(vertices)
    joints = flip_vertically(joints)

    joints_dict = {}
    counter = 0
    for idx in positions:
        joints_dict[idx] = joints[counter]
        counter += 1

    kdtree = KDTree(vertices)

    end_effectors = [node for node, degree in skeleton.degree() if degree == 1]
    nearest_vertices = {}
     # Cache vertex classes to avoid repeated lookups
    vertex_classes = {i: segmented_image[int(original_vertices[i][1])][int(original_vertices[i][0])] for i in range(len(original_vertices))}

    # Iterate over each end effector to find the nearest vertex and perform flood fill
    for end_effector in end_effectors:
        end_effector_pos = np.array(joints_dict[end_effector])
       
        # Find the nearest vertex using KDTree
        nearest_vertex_idx = kdtree.query(end_effector_pos)[1]

        # Get the class of the nearest vertex from the cache
        nearest_vertex_class = vertex_classes[nearest_vertex_idx]

        # Perform flood fill to find all vertices of the same class
        visited = set()
        queue = deque([nearest_vertex_idx])
        vertex_list = []

        while queue:
            current_idx = queue.popleft()  # Faster pop from the left
            if current_idx in visited:
                continue

            visited.add(current_idx)
            vertex_list.append(current_idx)

            # Check neighbors of the current vertex
            for face in faces:
                if current_idx in face:
                    for vertex_idx in face:
                        if vertex_idx not in visited and vertex_classes[vertex_idx] == nearest_vertex_class:
                            queue.append(vertex_idx)

        nearest_vertices[end_effector] = vertex_list
    
    # Find and assign remaining vertices to the nearest joint
    all_assigned_vertices = {v for verts in nearest_vertices.values() for v in verts}
    remaining_vertices = set(range(len(vertices))) - all_assigned_vertices

    # Create a KDTree for joints to find the nearest joint for remaining vertices
    joints_kdtree = KDTree(list(joints_dict.values()))

    for vertex_idx in remaining_vertices:
        vertex_pos = vertices[vertex_idx]
        nearest_joint_idx = joints_kdtree.query(vertex_pos)[1]
        nearest_joint = list(joints_dict.keys())[nearest_joint_idx]
        if nearest_joint in nearest_vertices:
            nearest_vertices[nearest_joint].append(vertex_idx)
        else:
            nearest_vertices[nearest_joint] = [vertex_idx]

    prototype_joint_to_vertices = {}

    for k in nearest_vertices:
        if joint_to_mapping.keys().__contains__(k):
            print(f"vertices for prototype joint: {joint_to_mapping[k]} : {nearest_vertices[k]}")
            prototype_joint_to_vertices[joint_to_mapping[k]] = nearest_vertices[k]

    

    return export_obj(vertices,faces,uvs,prototype_joint_to_vertices)