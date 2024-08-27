import cv2
import numpy as np
import pygltflib
import base64
import networkx as nx

def interpolate(v1, v2, val1, val2):
    if val1 != val2:
        return v1 + (v2 - v1) * (0.5 - val1) / (val2 - val1)
    else:
        return (v1 + v2) / 2

def marching_squares(image):
    rows, cols = image.shape
    vertices = []
    faces = []

    lookup_table = {
        1:  [(0,0), (0.5,0), (0,0.5), (0,0.5), (0.5,0), (0.5,0.5)],  # Bottom-left triangle
        2:  [(0.5,0), (1,0), (0.5,0.5), (0.5,0.5), (1,0), (1,0.5)],  # Bottom-right triangle
        4:  [(0.5,0.5), (1,0.5), (0.5,1), (0.5,1), (1,0.5), (1,1)],  # Top-right triangle
        8:  [(0,0.5), (0.5,0.5), (0,1), (0,1), (0.5,0.5), (0.5,1)],  # Top-left triangle
        3:  [(0,0), (1,0), (0,1), (1,0), (1,1), (0,1)],              # Vertical edge
        6:  [(0,0), (1,1), (1,0), (0,0), (0,1), (1,1)],              # Horizontal edge
        9:  [(0,0), (1,0), (1,1), (0,0), (1,1), (0,1)],              # Diagonal edge
        12: [(1,1), (0,0), (1,0), (0,1), (0,0), (1,1)],              # Diagonal edge
        5:  [(0,0), (0.5,0), (0.5,1), (0,0), (0.5,1), (0,1)],        # Left vertical split
        10: [(0.5,0), (1,0), (1,1), (0.5,0), (1,1), (0.5,1)],        # Right vertical split
        7:  [(0,0), (1,0), (0,1), (1,0), (1,1), (0,1)],              # Complex top-left, bottom-right
        14: [(0,0), (1,0), (1,1), (0,0), (1,1), (0,1)],              # Complex top-right, bottom-left
        13: [(1,0), (0,1), (1,1), (0,0), (0,1), (1,0)],              # Complex bottom-right, top-left
        11: [(1,0), (1,1), (0,1), (0,0), (0,1), (1,0)],              # Complex bottom-left, top-right
        15: [(0,0), (1,0), (1,1), (0,0), (1,1), (0,1)],              # Full square
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
                    face_vertices.append(vertices.index(v))
                for i in range(0, len(face_vertices), 3):
                    faces.append(face_vertices[i:i+3])
    return np.array(vertices), np.array(faces)

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

def export_gltf(vertices, faces, joints):
    points = np.zeros((vertices.shape[0], 3),dtype=np.float32)
    points[:,:2] = vertices.astype(np.float32)

    triangles = faces.astype(np.uint32)
    
    triangles_binary_blob = triangles.flatten().tobytes()
    points_binary_blob = points.tobytes()

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
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
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_INT,
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
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

    bones = []

    bones.append(pygltflib.Node(name="Root_Bone", translation=[0.0,0.0,0.0], children=[*range(1,len(joints)+1)]))
    
    for idx, joint in enumerate(joints):
        bone = pygltflib.Node(name=f"Bone_{idx}")
        bone.translation = [joint[0],joint[1],0.0]
        bones.append(bone)

    for bone in bones:
        gltf.nodes.append(bone)

    gltf.nodes.append(
        pygltflib.Node(
            name="Mesh",
            mesh=0
        )
    )

    binary_blob = triangles_binary_blob + points_binary_blob
    base64_blob = base64.b64encode(binary_blob).decode('utf-8')
    gltf.buffers[0].uri = f"data:application/octet_stream;base64,{base64_blob}"
    gltf.save("export.gltf")
    return "export.gltf"

def generate_mesh(image, skeleton):
    height, width = image.shape
    longer_side = height if height >= width else width
    downscale_factor = 80/longer_side


    image = cv2.resize(image, ((int)(width*downscale_factor),(int)(height*downscale_factor)), cv2.INTER_NEAREST)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary = binary/255

    positions = nx.get_node_attributes(skeleton, 'pos')
    joints = []
    for idx in positions:
        joints.append(positions[idx])
    for joint in joints:
        print(joint)
    for joint in joints:
        x = joint[0]
        y = joint[1]
        joint[0] = x*downscale_factor
        joint[1] = y*downscale_factor

    print(f"Image size: {image.shape}")
    for joint in joints:
        print(joint)
    joints = normalize_vertices(joints,longer_side*downscale_factor)
    vertices, faces = marching_squares(binary)
    vertices = normalize_vertices(vertices,longer_side*downscale_factor)
    vertices, joints = center_vertices(vertices, joints)
    vertices = flip_vertically(vertices)
    joints = flip_vertically(joints)

    return export_gltf(vertices,faces,joints)