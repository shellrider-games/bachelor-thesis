import cv2
import numpy as np

def generate_mesh(image, thickness=0.05):
    # Get image dimensions
    rows, cols = image.shape

    # Threshold the image to binary
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Create vertices for the grid points that are in white regions
    vertices = []
    vertex_indices = {}  # Dictionary to store the vertex indices

    for y in range(rows):
        for x in range(cols):
            if binary[y, x] == 255:
                vertex_indices[(x, y)] = len(vertices)
                vertices.append([x, y, 0.0])  # Add Z=0 for base layer

    # Create the top layer of vertices by offsetting the Z value by thickness
    top_vertices = [[x, y, thickness] for x, y, _ in vertices]
    vertices.extend(top_vertices)

    # Update the vertex indices for the top layer
    top_vertex_indices = {k: v + len(vertex_indices) for k, v in vertex_indices.items()}

    # Generate quads by connecting adjacent grid points
    faces = []
    for y in range(rows - 1):
        for x in range(cols - 1):
            if (x, y) in vertex_indices and (x + 1, y) in vertex_indices and (x, y + 1) in vertex_indices and (x + 1, y + 1) in vertex_indices:
                # Base layer face
                v0 = vertex_indices[(x, y)]
                v1 = vertex_indices[(x + 1, y)]
                v2 = vertex_indices[(x + 1, y + 1)]
                v3 = vertex_indices[(x, y + 1)]
                faces.append([v0, v1, v2, v3])

                # Top layer face
                tv0 = top_vertex_indices[(x, y)]
                tv1 = top_vertex_indices[(x + 1, y)]
                tv2 = top_vertex_indices[(x + 1, y + 1)]
                tv3 = top_vertex_indices[(x, y + 1)]
                faces.append([tv0, tv1, tv2, tv3])

                # Side faces
                faces.append([v0, v1, tv1, tv0])
                faces.append([v1, v2, tv2, tv1])
                faces.append([v2, v3, tv3, tv2])
                faces.append([v3, v0, tv0, tv3])

    # Convert vertices to NumPy array for further processing
    vertices_np = np.array(vertices)

    # Calculate the bounding box
    min_x, min_y, _ = np.min(vertices_np, axis=0)
    max_x, max_y, _ = np.max(vertices_np, axis=0)

    # Calculate the center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Calculate the scale factor to fit the mesh to size 1
    max_dim = max(max_x - min_x, max_y - min_y)
    scale_factor = 1.0 / max_dim

    obj_text = ''

    
    # Write vertices and UVs
    for vertex in vertices_np:
        flipped_x = vertex[0]  # No need to flip X axis
        transformed_x = (flipped_x - center_x) * scale_factor
        transformed_y = (vertex[1] - center_y) * scale_factor
        transformed_y = -transformed_y  # Flip Y axis to correct orientation
        transformed_z = vertex[2]  # Do not scale the Z coordinate
        u = vertex[0] / cols
        v = 1.0 - (vertex[1] / rows)  # Flip the V coordinate
        obj_text += f"v {transformed_x} {transformed_y} {transformed_z}\n"
        obj_text += f"vt {u} {v}\n"
    
    # Write faces with UVs
    for face in faces:
       obj_text += f"f {face[0] + 1}/{face[0] + 1} {face[1] + 1}/{face[1] + 1} {face[2] + 1}/{face[2] + 1} {face[3] + 1}/{face[3] + 1}\n"
    return obj_text