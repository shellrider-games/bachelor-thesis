# This file includes the USSR class, that creates an USSR without CEF.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage import measure
from skimage.draw import circle_perimeter as circle
from skimage.morphology import dilation
from skimage.transform import resize


def ray_line(origin, ray, p1, p2):
    # https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
    v1 = origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray[1], ray[0]])
    dotv2v3 = np.dot(v2, v3)
    if np.abs(dotv2v3) < 0.00001:
        return None
    t1 = np.abs(np.cross(v2, v1)) / dotv2v3
    t2 = np.dot(v1, v3) / dotv2v3
    if t1 < 0 or t2 < 0 or t2 > 1:
        return None
    return origin + ray * t1


ECC_DIRECTIONS = np.array([[1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]])


def eccentricity(img):
    graph = nx.Graph()
    shape_parts = np.array(np.where(img == 1)).T
    for pixel in shape_parts:
        for direction in ECC_DIRECTIONS:
            pixel2 = pixel + direction
            try:
                if img[tuple(pixel2)]:
                    graph.add_edge(tuple(pixel), tuple(pixel2))
            except:
                print('Warning:', pixel2, 'outside of the image')
    result = np.zeros_like(img).astype(int)
    ecc = nx.eccentricity(graph)
    for ind in ecc.keys():
        result[tuple(ind)] = ecc[ind]
    return result


def eccentricity_speed(img):
    graph = nx.Graph()
    shape_parts = np.array(np.where(img == 1)).T
    for pixel in shape_parts:
        for direction in ECC_DIRECTIONS:
            pixel2 = pixel + direction
            if img[tuple(pixel2)]:
                graph.add_edge(tuple(pixel), tuple(pixel2), weight=1)

    while True:
        change = False
        for node in graph.node:
            if graph.degree(node) == 2:
                combine = graph.neighbors(node)
                edge_weight = graph[node][combine[0]]["weight"] + graph[node][combine[1]]["weight"]
                graph.add_edge(*combine, weight=edge_weight)
                graph.remove_node(node)
                change = True
                break
        if change is False:
            break

    result = np.zeros_like(img).astype(int)
    ecc = nx.eccentricity(graph)
    for ind in ecc.keys():
        result[tuple(ind)] = ecc[ind]
    return dilation(result)


def us_resize(us, size):
    """
    Return an us resized to size sample points
    """
    return resize(us.reshape(1, len(us)), (1, size), mode="wrap")[0]  # This will trunc at 1


def us_rescale(us):
    """
    Return an us rescaled between 0 and 1
    """
    return us.astype(float) / us.max()


def us_normalize(us, size):
    """
    Simple Normalization of us
    """
    return us_rescale(us_resize(us, size))


def special_point_resize(points, length, size):
    """
    Re-index special points of us with length "length" to "size"
    """
    return np.round(points.astype(float) / length * size).astype(int)


def us_sal_normalize(us, points, spb, skip=0, ecc=None):
    """
    Applies structure aware length normalization with points and size per branch (spb)
    if skip > 0, return a tupel of normaliziation and new special points and skipping all branches
    smaller than skip
    """
    length = len(us)
    result = []
    last_p = -len(us) + points[-1]
    us_scaled = us.copy()
    sp_points = []
    if ecc is None:
        ecc = sp_points
    sp_ecc = []
    i = -1
    for p in points:
        i += 1
        if p - last_p < skip:
            last_p = p
            continue
        sp_points.append(p)
        sp_ecc.append(ecc[i])
        resize_param = int(spb / (p - last_p) * length)
        resized = us_resize(us_scaled, resize_param)
        indices = special_point_resize(np.array([last_p, p]), length, resize_param)
        result = result + resized[indices[0]:indices[1]].tolist()
        last_p = p

    if skip > 0:
        if ecc is not None:
            return us_rescale(np.array(result)), np.array(sp_points), np.array(sp_ecc)
        return us_rescale(np.array(result)), np.array(sp_points)
    return us_rescale(np.array(result))


def us_sal_renormalize_startpoint(us, ecc_points, spb):
    """
    Redo normalization based on new (reduced) points eccentricity. Points are consided to have same
    distance spb in us vector
    """
    smallest = np.where(ecc_points == ecc_points.min())[0]
    startpoint_special = smallest[np.argmin((ecc_points[np.mod(smallest + 1, len(ecc_points))]))]
    startpoint = startpoint_special * spb
    return np.hstack([us[startpoint:], us[:startpoint]])


def rotateVector(vector, pos):
    return np.array(vector[pos:].tolist() + vector[:pos].tolist())


class USSR:
    # always turn left
    NEXTDIRECTION = {
        (1, 0): [1, 1],
        (1, 1): [0, 1],
        (0, 1): [-1, 1],
        (-1, 1): [-1, 0],
        (-1, 0): [-1, -1],
        (-1, -1): [0, -1],
        (0, -1): [1, -1],
        (1, -1): [1, 0]
    }

    RIGHTDIRECTION = {
        (1, 0): [0, -1],
        (1, 1): [1, -1],
        (0, 1): [1, 0],
        (-1, 1): [1, 1],
        (-1, 0): [0, 1],
        (-1, -1): [-1, 1],
        (0, -1): [-1, 0],
        (1, -1): [-1, -1]
    }
    """
    # Always turn right
    NEXTDIRECTION = {
        (1, 0): [1, -1],
        (1, -1): [0, -1],
        (0, -1): [-1, -1],
        (-1, -1): [-1, 0],
        (-1, 0): [-1, 1],
        (-1, 1): [0, 1],
        (0, 1): [1, 1],
        (1, 1): [1, 0]
    }

    RIGHTDIRECTION = {
        (1, 0): [0, 1],
        (1, 1): [-1, 1],
        (0, 1): [-1, 0],
        (-1, 1): [-1, -1],
        (-1, 0): [0, -1],
        (-1, -1): [1, -1],
        (0, -1): [1, 0],
        (1, -1): [1, 1]
    }
    """

    def __init__(self, sequence=None, shape=None, skeleton=None):
        """
        Creates an USSR
        :param sequence: a np array with N elements corresponding to the radii on the ussr
        :param shape: the shape as 2d binary image with white pixel being shape and black background
        :param skeleton: 2d binary image of skeleton or Skeleton object
        """
        self.sequence = sequence
        self.shape = shape
        self.skeleton = skeleton
        self.polygon = None
        self.dt = None
        self.matdt = None
        self.normals = None
        self.shape_polygon = None
        self.voronoi = None
        self.skel_eccentricity = None
        self.length_on_polygon = None

    @classmethod
    def from_shape(cls, shape, skeleton=None, thres=None):
        """
        Create USSR from a shape
        :param shape: shape as 2d binary image with white pixel being shape
        :param skeleton: 2d binary image of skeleton or Skeleton object
        :return: USSR object
        """
        if skeleton is None:
            from voronoi_skeleton import skeletonize
            skeleton = skeletonize(shape, tresh=thres)
        ussr = USSR(shape=shape, skeleton=skeleton)
        ussr.build_polygon()
        ussr.build_shape_polygon(0)
        ussr.get_normals(smooth=True)
        return ussr

    def build_shape_polygon(self, tolerance=0):
        contours = measure.find_contours(self.shape.astype(int), self.shape.astype(int).max() / 2)
        points = np.concatenate(contours)

        # _, contours, _ = cv2.findContours(self.shape.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contarr = np.array(contours[0]).T
        # contarr = np.array([contarr[1][0], contarr[0][0]])
        # points = contarr.T

        if tolerance > 0:
            from skimage.measure import approximate_polygon
            points = approximate_polygon(points, tolerance)
        self.shape_polygon = points

    def build_polygon(self):
        assert self.shape is not None and self.skeleton is not None
        assert type(self.shape) == np.ndarray and len(self.shape.shape) == 2
        assert type(self.skeleton) == np.ndarray and len(self.skeleton.shape) == 2

        polygon = None
        from mahotas.morph import hitmiss
        # Start at some random skeleton end point and always go right around the whole tree, at some point we should
        # again reach that point
        pattern1 = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [2, 1, 2]])
        pattern2 = np.array([[0, 0, 0],
                             [0, 1, 2],
                             [0, 2, 1]])

        patterns = [pattern1, np.rot90(pattern1), np.rot90(pattern1, 2), np.rot90(pattern1, 3),
                    pattern2, np.rot90(pattern2), np.rot90(pattern2, 2), np.rot90(pattern2, 3)]
        endpoints_img = np.zeros_like(self.skeleton).astype(int)
        for pattern in patterns:
            endpoints_img += hitmiss(self.skeleton, pattern)
        endpoints = [tuple(p) for p in np.array(np.where((endpoints_img) >= 1)).T]

        pattern1 = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
        pattern2 = np.array([[1, 0, 1],
                             [0, 1, 0],
                             [1, 0, 1]])
        pattern3 = np.array([[1, 0, 1],
                             [0, 1, 2],
                             [1, 2, 0]])
        pattern4 = np.array([[0, 1, 2],
                             [1, 1, 0],
                             [0, 1, 2]])
        pattern5 = np.array([[2, 1, 2],
                             [1, 1, 0],
                             [2, 0, 1]])
        pattern6 = np.array([[1, 0, 1],
                             [0, 1, 0],
                             [2, 1, 2]])

        patterns = [pattern1, pattern2,
                    pattern3, np.rot90(pattern3), np.rot90(pattern3, 2), np.rot90(pattern3, 3),
                    pattern4, np.rot90(pattern4), np.rot90(pattern4, 2), np.rot90(pattern4, 3),
                    pattern5, np.rot90(pattern5), np.rot90(pattern5, 2), np.rot90(pattern5, 3),
                    pattern6, np.rot90(pattern6), np.rot90(pattern6, 2), np.rot90(pattern6, 3)]
        branchpoints_img = np.zeros_like(self.skeleton).astype(int)
        for pattern in patterns:
            branchpoints_img += hitmiss(self.skeleton, pattern)
        branchpoints = [tuple(p) for p in np.array(np.where((branchpoints_img) >= 1)).T]

        if len(endpoints) <= 0:
            import ipdb;
            ipdb.set_trace()
        startpoint = np.array(endpoints[0])
        nextpoint = self.skeleton[startpoint[0] - 1:startpoint[0] + 2, startpoint[1] - 1:startpoint[1] + 2].copy()
        nextpoint[1, 1] = 0
        nextpoint = np.array(np.where((nextpoint) == 1)).transpose()[0] - 1

        # Find next point, starting with direction [0]
        # nextpoint = None
        # direction = direction = np.array([1,1])
        # import pdb;pdb.set_trace()
        # while True:
        #    try:
        #        if self.skeleton[tuple(startpoint + direction)] == 1:
        #            nextpoint = startpoint + direction
        #            break
        #    except IndexError:
        #        # It might happen that our skeleton is on the border of the image... just skip pixel outside the image.
        #        pass
        #    direction = np.array(self.NEXTDIRECTION[tuple(direction)])

        nextpoint = startpoint + nextpoint
        polygon = [startpoint]
        current_point = nextpoint
        direction = nextpoint - startpoint
        # print("Build Polygon",flush=True)
        index = 1
        self.branchpoints = []
        self.endpoints = []
        # if tuple(startpoint) in branchpoints:
        #    self.branchpoints.append(0)
        # else:
        self.endpoints.append(0)

        length_on_polygon = [0]

        last_branch = False
        while True:
            if (current_point == startpoint).all():
                break

            if tuple(current_point) in endpoints:
                self.endpoints.append(index)
                polygon.append(current_point)
            else:
                # Offset from skeleton center, so we have unique distances
                polygon.append(current_point + np.array(self.RIGHTDIRECTION[tuple(direction)]) * 0.3)

            length_on_polygon.append(np.linalg.norm(direction))
            if tuple(current_point) in branchpoints and last_branch is False:
                self.branchpoints.append(index)
                last_branch = True
            else:
                last_branch = False

            direction = np.array(self.NEXTDIRECTION[tuple(direction * -1)])
            while True:
                try:
                    if self.skeleton[tuple(current_point + direction)] == 1:
                        nextpoint = current_point + direction

                        # It can happen, that we skip a branchpoint here, so check neighborhood
                        if tuple(current_point) not in branchpoints:

                            opt1 = np.array([current_point[0], nextpoint[1]])
                            opt2 = np.array([nextpoint[0], current_point[1]])

                            if (opt1 != nextpoint).any() and (opt1 != current_point).any():
                                if branchpoints_img[tuple(opt1)] > 0:
                                    polygon.append(opt1 + np.array(self.RIGHTDIRECTION[tuple(direction)]) * 0.3)
                                    length_on_polygon.append(np.linalg.norm(direction))
                                    index += 1
                                    if not last_branch:
                                        self.branchpoints.append(index)
                                        last_branch = True

                            if (opt2 != nextpoint).any() and (opt2 != current_point).any():
                                if branchpoints_img[tuple(opt2)] > 0:
                                    polygon.append(opt2 + np.array(self.RIGHTDIRECTION[tuple(direction)]) * 0.3)
                                    length_on_polygon.append(np.linalg.norm(direction))
                                    index += 1
                                    if not last_branch:
                                        self.branchpoints.append(index)
                                        last_branch = True

                        current_point = nextpoint

                        break
                except IndexError:
                    # It might happen that our skeleton is on the border of the image... just skip pixel outside the image.
                    pass
                direction = np.array(self.NEXTDIRECTION[tuple(direction)])

            index += 1

        self.polygon = np.array(polygon)
        self.length_on_polygon = np.cumsum(length_on_polygon)
        """
        # Now we rotate the polygon point list, so the first point is the branching point with the smallest eccentricity
        ecc = eccentricity(self.skeleton) * (branchpoints_img + endpoints_img).astype(bool).astype(int)
        ecc = ecc.astype(float)
        ecc[ecc == 0] = np.inf
        smallest_ecc = np.array(np.unravel_index(np.argmin(ecc), ecc.shape))
        points = []
        for i, v in enumerate(np.round(self.polygon).astype(int)):
            if (v == smallest_ecc).all():
                points.append(i)
        startpoint = points[0]
        for p in points[1:]:
            if self.polygon[startpoint][0] * 1.5 + self.polygon[startpoint][1] < self.polygon[p][0] * 1.5 + \
                    self.polygon[p][1]:
                startpoint = p
        """
        self.endpoints = np.array(self.endpoints)
        self.branchpoints = np.array(self.branchpoints)
        self.skel_eccentricity = eccentricity(self.skeleton)
        self.rotate_to_eccentricity_minimum()
        # self.rotate_polygon_point_list(startpoint)
        # import pdb;pdb.set_trace()

    def flip(self):
        """
        Flip the us vector, as if walked in other direction
        NOT WORKING!
        :return:
        """
        self.skeleton = np.flip(self.skeleton, axis=1)
        self.polygon = np.flip(self.polygon, axis=0)
        self.branchpoints = len(self.polygon) - 1 - self.branchpoints
        self.endpoints = len(self.polygon) - 1 - self.endpoints
        self.rotate_to_eccentricity_minimum()

    def rotate_to_eccentricity_minimum(self, thresh=0.05):
        """
        Rotates the us and end/branchpoints to minimum eccentricity.
        :param thresh: relative thresh from max eccentricity
        :return:
        """
        skel_ecc = self.skel_eccentricity
        special_points = (self.branchpoints.tolist() + self.endpoints.tolist())
        special_points.sort()
        us_special_ecc = skel_ecc[tuple(np.round(self.polygon[special_points]).astype(int).T)]
        thresh = thresh * us_special_ecc.max()
        smallest = np.where(us_special_ecc < us_special_ecc.min() + thresh)[0]

        # import pdb;pdb.set_trace()

        # possible rotations
        check = np.array([rotateVector(us_special_ecc, s) for s in smallest])
        check[:, 0] = np.arange(len(check))

        startpoint = None;

        for i in range(1, len(us_special_ecc)):
            values = np.array([c[i] for c in check])
            smallest_o = np.where(values < values.min() + thresh)[0]
            if len(smallest_o) == 1:
                startpoint = smallest[check[smallest_o[0]][0]]
                break
            check = check[smallest_o]

        if startpoint is None:
            startpoint = smallest[np.argmin((us_special_ecc[np.mod(smallest + 1, len(us_special_ecc))]))]

        startpoint = special_points[startpoint]

        """
        smallest = np.where(us_special_ecc == us_special_ecc.min())[0]
        startpoint_special = smallest[np.argmin((us_special_ecc[np.mod(smallest + 1, len(us_special_ecc))]))]
        startpoint = special_points[startpoint_special]
        """
        self.rotate_polygon_point_list(startpoint)

    def rotate_to_max_radius(self, thresh=0.05):
        if self.dt is None:
            self.dt = distance_transform_edt(self.shape)
        skel_ecc = self.dt
        special_points = (self.branchpoints.tolist() + self.endpoints.tolist())
        special_points.sort()
        us_special_ecc = skel_ecc[tuple(np.round(self.polygon[special_points]).astype(int).T)]
        thresh = thresh * us_special_ecc.max()
        smallest = np.where(us_special_ecc >= us_special_ecc.max() - thresh)[0]

        # import ipdb;ipdb.set_trace()

        # possible rotations
        check = np.array([rotateVector(us_special_ecc, s) for s in smallest])
        check[:, 0] = np.arange(len(check))

        startpoint = None;

        for i in range(1, len(us_special_ecc)):
            values = np.array([c[i] for c in check])
            smallest_o = np.where(values >= values.max() - thresh)[0]
            if len(smallest_o) == 1:
                startpoint = smallest[int(check[smallest_o[0]][0])]
                break
            check = check[smallest_o]

        if startpoint is None:
            startpoint = smallest[np.argmax((us_special_ecc[np.mod(smallest + 1, len(us_special_ecc))]))]

        startpoint = special_points[startpoint]

        """
        smallest = np.where(us_special_ecc == us_special_ecc.min())[0]
        startpoint_special = smallest[np.argmin((us_special_ecc[np.mod(smallest + 1, len(us_special_ecc))]))]
        startpoint = special_points[startpoint_special]
        """
        self.rotate_polygon_point_list(startpoint)

    def rotate_polygon_point_list(self, number):
        """
        Rotate the points "number" of times to the left
        :param number:
        :return:
        """
        self.polygon = np.array(self.polygon[number:].tolist() + self.polygon[:number].tolist())
        self.endpoints = (self.endpoints - number) % len(self.polygon)
        self.branchpoints = (self.branchpoints - number) % len(self.polygon)

    def get_branch_points(self):
        return self.branchpoints

    def get_end_points(self):
        return self.endpoints

    def get_normals(self, smooth=False):
        edge_normals = []
        for i in range(len(self.polygon)):
            # i0 = self.polygon[(i -2) % len(self.polygon)]
            # i1 = self.polygon[(i -1) % len(self.polygon)]
            # i2 = self.polygon[i]
            # i3 = self.polygon[(i +1) % len(self.polygon)]
            # i4 = self.polygon[(i +2) % len(self.polygon)]
            # Use 5 point interpolytion
            # tan = i4-i0
            # normal = [-tan[1], tan[0]]
            # normal = normal/np.linalg.norm(normal)
            # normals.append(normal)
            tangent = self.polygon[i] - self.polygon[(i - 1) % len(self.polygon)]
            normal = [-tangent[1], tangent[0]]
            normal = normal / np.linalg.norm(normal)
            edge_normals.append(normal)

        self.edge_normals = np.array(edge_normals)

        vertex_normals = []
        for i in range(len(self.polygon)):
            vertex_normals.append((self.edge_normals[i] + self.edge_normals[(i + 1) % len(self.polygon)]) / 2)

        self.normals = -np.array(vertex_normals)
        if smooth:
            from skimage.filters import gaussian
            smooth_normals = gaussian(self.normals, sigma=5, mode="wrap")
            norm = np.linalg.norm(smooth_normals, axis=1)
            self.normals = smooth_normals / np.tile(norm, (2, 1)).T

    def closest_intersection_ray_shape(self, p, ray):
        intersections = []
        for i in range(len(self.shape_polygon)):
            p1 = self.shape_polygon[i]
            p2 = self.shape_polygon[(i + 1) % len(self.shape_polygon)]
            intersect = ray_line(p, ray, p1, p2)
            if intersect is not None:
                intersections.append(intersect)
        if len(intersections) == 0:
            return None
        intersections = np.array(intersections)
        return intersections[cdist([p], intersections).argmin()]

    def get_coordinate_images(self, normalize=True, return_over_shape=False, circ_sqrt=False):
        """
        Returns the coordinate in radiants, and sp as MATDT/(MATDT+DT) for the whole image space
        :param return_over_shape: give distance coordinate that are outside the shape...
        :return:
        """
        if self.dt is None:
            self.dt = distance_transform_edt(self.shape)
        if return_over_shape:
            dt = self.dt - (distance_transform_edt(np.invert(self.shape)) - 1)
        else:
            dt = self.dt
        if self.matdt is None:
            self.matdt = distance_transform_edt(np.invert(self.skeleton))

        if normalize:
            sp = self.matdt / (self.matdt + dt)
        else:
            sp = self.matdt

        if self.voronoi is None:
            xx, yy = np.mgrid[:self.shape.shape[0], :self.shape.shape[1]]
            xy = np.vstack([xx.ravel(), yy.ravel()]).T
            dists = cdist(xy, self.polygon)  # , metric="cityblock")
            xy_dists = np.argmin(dists, axis=1)
            self.voronoi = xy_dists.reshape(self.shape.shape)
            """
            size = self.shape.shape
            distance = np.ones(np.array(size) * 2)
            distance[size] = 0
            distance = distance_transform_edt(distance)
            mdmap = np.zeros(size)
            idmap = np.ones(size)
            i = 1
            for p in self.polygon:
                tmp = distance[int(size[0] - p[0]):int(size[0] - p[0] + size[0]),
                      int(size[1] - p[1]):int(size[1] - p[1] + size[1])]
                if i == 1:
                    mdmap = tmp.copy()
                    i += 1
                    continue
                diff = mdmap - tmp
                idmap[diff > 0] = i
                mdmap = np.minimum(mdmap, tmp)
                i += 1
            self.voronoi = idmap - 1
            """

        circular = self.voronoi / len(self.polygon) * np.pi * 2
        if circ_sqrt:
            # Normalize circular with sqrt(2) for diagonal
            circular_out = self.length_on_polygon[self.voronoi] / self.length_on_polygon.max() * np.pi * 2
            return circular, sp, circular_out

        return circular, sp

    def get_coordinate_imgaes_new(self, normalize=False, dist_divide=None, circ_sqrt=False):
        print("Coordstuff")
        if circ_sqrt:
            circulae, sp, circulae_out = self.get_coordinate_images(normalize=normalize, circ_sqrt=circ_sqrt)
        else:
            circulae, sp = self.get_coordinate_images(normalize=normalize, circ_sqrt=circ_sqrt)

        if dist_divide is not None:
            # TODO
            sp = np.round(sp * dist_divide, 0).astype(float) / dist_divide
        sp[~self.shape] = np.nan
        print("--- other")

        delta = np.full(sp.shape, 0.5)
        normback = ((2 * np.pi) / len(self.polygon))

        unique_cir = np.unique(circulae)
        for uni in unique_cir:
            uniques, counts = np.unique(sp[circulae == uni], return_counts=True)
            if counts.max() > 1:
                distances = sp[circulae == uni]
                relevant_parts = np.array(np.where((circulae == uni) == 1)).T
                polygon_part = int(np.round(uni / normback))
                # tangent = (np.round(self.polygon[(polygon_part-1)%len(self.polygon)])-np.round(self.polygon[(polygon_part+1)%len(self.polygon)])).astype(int)
                # tangent = tangent/np.linalg.norm(tangent)
                # tangent = np.vstack([tangent]*len(relevant_parts))
                normal = self.normals[polygon_part]
                curr_pos = np.round(self.polygon[polygon_part]).astype(int)
                diff = relevant_parts - curr_pos
                # import ipdb;ipdb.set_trace()
                # diff_n = diff/np.vstack([np.linalg.norm(diff,axis=1)]*2).T
                # angles = np.cos((tangent*diff_n).sum(axis=1))
                c = np.dot(normal, [1, 0])
                s = np.sin(np.arccos(c))
                rot_mat = np.array([[c, -s], [s, c]])
                diff = diff @ rot_mat
                angles = np.arctan2(diff[:, 0], diff[:, 1])
                for u in uniques:
                    # Here we go through all distances an calculate delta
                    relevant_angles = angles[distances == u]
                    if len(relevant_angles) < 2:
                        continue
                    mmax = relevant_angles.max()
                    mmin = relevant_angles.min()
                    deltas = (relevant_angles - mmin) / (mmax - mmin)
                    delta[tuple(relevant_parts[distances == u].T)] = deltas

        print("WHOA!")
        delta = np.nan_to_num(delta, 0.5)
        if circ_sqrt:
            return circulae_out, sp, delta
        return circulae, sp, delta

    def get_coordinate_imgaes_new_skel_apporach(self, normalize=False, dist_divide=None, tau=2, circ_sqrt=False):
        """
        Get delta by skeleton approach
        :param normalize:
        :param dist_divide:
        :return:
        """
        print("Coordstuff Skel approach")
        circulae, sp = self.get_coordinate_images(normalize=normalize, circ_sqrt=circ_sqrt)

        if dist_divide is not None:
            # TODO
            sp = np.round(sp * dist_divide, 0).astype(float) / dist_divide
        sp[~self.shape] = np.nan
        print("--- other")

        delta = np.full(sp.shape, 0.5)
        normback = ((2 * np.pi) / len(self.polygon))

        unique_cir = np.unique(circulae)
        for uni in unique_cir:
            uniques, counts = np.unique(sp[circulae == uni], return_counts=True)
            if counts.max() > 5:
                distances = sp[circulae == uni]
                relevant_parts = np.array(np.where((circulae == uni) == 1)).T
                polygon_part = int(np.round(uni / normback))

                neg_ind = (polygon_part - tau) % len(self.polygon)
                pos_ind = (polygon_part + tau) % len(self.polygon)
                curr_pos = np.round(self.polygon[polygon_part])
                diff_neg = curr_pos - np.round(self.polygon[neg_ind])
                diff_pos = np.round(self.polygon[pos_ind]) - curr_pos
                norm_neg = np.array([diff_neg[1], -diff_neg[0]])
                norm_pos = np.array([diff_pos[1], -diff_pos[0]])
                norm_neg = norm_neg / np.linalg.norm(norm_neg)
                norm_pos = norm_pos / np.linalg.norm(norm_pos)

                alpha1 = np.arccos(np.dot(norm_neg, norm_pos))

                diff = relevant_parts - curr_pos
                diff_n = diff / np.vstack([np.linalg.norm(diff, axis=1)] * 2).T
                angles = np.arccos((diff_n * norm_neg).sum(axis=1))

                delta[tuple(relevant_parts.T)] = np.maximum(np.minimum(angles / alpha1, 1), 0)

        print("WHOA!")
        delta = np.nan_to_num(delta, 0.5)
        return circulae, sp, delta

    def get_coordinate(self, xy, return_closest=False):
        """
        Return circular c, distance d and angular a component coordinate
        :param xy: point
        :param return_closest: return a tupel with (cda, closest, normal) with closest being the closest point on skeleton
        :return:
        """
        # Get closest point on polygon
        closest = cdist([xy], self.polygon).argmin()
        circular = closest / len(self.polygon) * np.pi * 2

        # Get distance and circular value
        vector = self.polygon[closest] - xy
        distance = np.linalg.norm(vector)

        # Get angular component from normal vector
        vector = vector / np.linalg.norm(vector)
        # import ipdb;ipdb.set_trace()
        angular = np.arccos(np.dot(vector, self.normals[closest])) * np.sign(np.cross(vector, self.normals[closest]))

        if return_closest:
            return (np.array([circular, distance, angular]), self.polygon[closest].astype(int), self.normals[closest])

        return np.array([circular, distance, angular])

    def get_xy_coordinate(self, cda):
        """
        Return x,y coordinate of c,d,a coordinate
        :param cda:
        :return:
        """
        pass

    def ussr_from_normal(self, normalize=None):
        ussr = []
        for i in range(len(self.polygon)):
            closest = self.closest_intersection_ray_shape(self.polygon[i], self.normals[i])
            ussr.append(np.linalg.norm(closest - self.polygon[i]))
        if normalize is None or type(normalize) != int:
            return np.array(ussr)
        else:
            return self.normalize_us(np.array(ussr), normalize)

    def get_special_points(self):
        """
        Returns combined list of position of branching and endpoints in order of visiting
        (each branch point is included thrice, each endpoint once)
        :return:
        """
        special_points = (self.branchpoints.tolist() + self.endpoints.tolist())
        special_points.sort()
        return np.array(special_points)

    def normalize_us(self, us, normalize):
        us = us.reshape((1, len(us)))
        # norm_us = imresize(us, (1, normalize))[0]
        # Trick: repeat 3 times, resize and cut out the middle (this leaves better interpolation?)
        maxval = us.max()
        us = us / maxval
        norm_us = resize(us, (1, normalize), mode="wrap")[0]  # This one is picky... It will trunc at 1
        minval = norm_us.min()
        norm_us -= minval
        norm_us = norm_us.astype(float)
        # maxval = norm_us.max()
        # norm_us /= maxval
        # ind_max = norm_us.argmax()
        # norm_us = np.array(norm_us[ind_max:].tolist() + norm_us[:ind_max].tolist())
        # return norm_us, [ind_max, minval, maxval]
        return norm_us, [minval, maxval]

    def get_unraveled_skeleton(self, normalize=None):
        """
        Returns a unraveled skeleton feature vector of length of the whole unraveled skeleton
        :param normalize: if this is not None and int, the vector gets scaled to this length,
                          additionally the values are normalized between 0 and 1 and circled so the highest value is
                          in first place
        :return:
        """
        assert self.shape is not None and self.skeleton is not None
        assert type(self.shape) == np.ndarray and len(self.shape.shape) == 2
        assert type(self.polygon) == np.ndarray

        if self.dt is None:
            self.dt = distance_transform_edt(self.shape)
        if normalize is None or type(normalize) != int:
            return self.dt[tuple(np.round(self.polygon.T).astype(int))]
        else:
            us = self.dt[tuple(np.round(self.polygon.T).astype(int))]
            return self.normalize_us(us, normalize)

    def get_sal_normalized_us(self, size=100, spb=20, askip=0, rskip=0):
        """
        return the us vector as a structure aware length normalization with possible small length skipping
        :param size: size of resulting vector
        :param spb: samples per branch on intermediate vector.
        :param askip: absolute branch length skipping
        :param rskip: relateve skip to total skeleton length (1=full skeleton) (if rskip > 0 askip is ignored)
        :return: tuple: (us, special_points)
        """
        us = self.get_unraveled_skeleton()
        sp = self.get_special_points()
        if rskip > 0:
            askip = rskip * len(us)
        us_saln, new_sp = us_sal_normalize(us, sp, spb, skip=askip)
        point_ecc = self.get_eccentricity_on_special_points()
        us_saln = us_sal_renormalize_startpoint(us_saln, point_ecc, spb)
        us_normalized = us_normalize(us_saln, size)
        us_sp = (np.arange(0, size + 1, size / len(new_sp))).astype(int)
        return us_normalized, us_sp

    def get_normalized_us(self, size=100, askip=0, rskip=0):
        """
        return the us vector as a simple normalization with possible small length skipping
        :param size: size of resulting vector
        :param askip: absolute branch length skipping
        :param rskip: relateve skip to total skeleton length (1=full skeleton) (if rskip > 0 askip is ignored)
        :return: tuple: (us, special_points)
        """
        us = self.get_unraveled_skeleton()
        sp = self.get_special_points()
        if rskip > 0:
            askip = rskip * len(us)

        # point_ecc = self.skel_eccentricity[tuple(np.round(self.polygon[new_sp]).astype(int).T)]

        us_normalized = us_normalize(us, size)
        us_sp = np.round(sp.astype(float) / len(us) * size).astype(int)
        return us_normalized, us_sp

    def reconstruct(self, us, norm_params=(0, 0, 1)):
        us += norm_params[1]
        us *= norm_params[2]
        ind_max = int(norm_params[0])
        us = np.array(us[-ind_max:].tolist() + us[:-ind_max].tolist())
        result = np.zeros_like(self.shape)
        for i in range(len(self.polygon)):
            index_us = float(i * len(us)) / len(self.polygon)
            t = index_us - int(index_us)
            us_val = us[int(index_us)] * (1 - t) + us[(int(index_us) + 1) % len(us)] * (t)
            # import ipdb;ipdb.set_trace()
            rr, cc = circle(int(self.polygon[i][0]), int(self.polygon[i][1]), max(1, int(us_val)))
            try:
                rr = np.minimum(rr, result.shape[0] - 1)
                cc = np.minimum(cc, result.shape[1] - 1)
                rr = np.maximum(rr, 0)
                cc = np.maximum(cc, 0)
                result[rr, cc] = 1
            except:
                import ipdb;
                ipdb.set_trace()
        return result

    def reconstruct_with_normals(self, us, norm_params=(0, 0, 1)):
        us += norm_params[1]
        us *= norm_params[2]
        ind_max = int(norm_params[0])
        us = np.array(us[-ind_max:].tolist() + us[:-ind_max].tolist())
        result = np.zeros_like(self.shape)
        for i in range(len(self.polygon)):
            index_us = i / len(self.polygon) * len(us)
            t = index_us - int(index_us)
            us_val = us[int(index_us)] * (1 - t) + us[(int(index_us) + 1) % len(us)] * t
            shiftet = self.polygon[i] + self.normals[i] * us_val
            if shiftet[0] >= result.shape[0] or shiftet[1] >= result.shape[1]:
                continue
            result[tuple(shiftet.astype(int))] = 1
        return result

    #### ----- DEBUG FUNCS ----- ####

    def plot(self, *args, **kwargs):
        for i in range(len(self.polygon)):
            plt.plot([self.polygon[i][1], self.polygon[(i + 1) % len(self.polygon)][1]],
                     [self.polygon[i][0], self.polygon[(i + 1) % len(self.polygon)][0]], *args, **kwargs)

    def get_eccentricity_on_special_points(self, points=None):
        if points is None:
            points = self.get_special_points()
        return self.skel_eccentricity[tuple(np.round(self.polygon[points]).astype(int).T)]
