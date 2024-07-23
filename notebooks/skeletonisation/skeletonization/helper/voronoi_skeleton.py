import numpy as np
from scipy.spatial import Voronoi
from skimage import draw
from skimage import measure
from skimage.morphology import skeletonize as skeletonize2


def voronoi_skeletonize(img, residual_type=1):
    """
    Creates a voronoi tessalation and gives every edge a residual value
    :param img: black background, white foreground
    :param residual_type: 1: Circularity Residual, 2: Bi-Circularity Residual, 3: Chord Residual
    :return:
    """
    contours = measure.find_contours(img, img.max() / 2)
    contour = np.concatenate(contours)
    # Build up boundary potential function
    boundary_potential = []
    boundary_lengths = []
    for cont in contours:
        length = 0
        last_point = None
        for point in cont:
            if last_point is None:
                last_point = point
            else:
                length += 1  # np.linalg.norm(point-last_point)
            boundary_potential.append(length)
        boundary_lengths = boundary_lengths + [length] * len(cont)

    boundary_potential = np.array(boundary_potential)
    boundary_lengths = np.array(boundary_lengths)

    # for i in range(1,len(contour)):
    #    plt.plot([contour[i-1][1],contour[i][1]],[contour[i-1][0],contour[i][0]],"-",
    #             color=hsv_to_rgb([0,0,boundary_potential[i]/boundary_lengths[i]]))

    # plt.show()
    # import ipdb;ipdb.set_trace()

    vor = Voronoi(contour)
    # import ipdb;ipdb.set_trace()
    # Get all contour point pairs that share an edge in voronoi
    residual_points = np.concatenate([vor.points[vor.ridge_points[:, 0]], vor.points[vor.ridge_points[:, 1]]], axis=1)
    ridge_vertices = np.array(vor.ridge_vertices)
    ridge_vertices = np.concatenate([vor.vertices[ridge_vertices[:, 0]], vor.vertices[ridge_vertices[:, 1]]], axis=1)
    # Calculate the residual
    if residual_type == 1 or residual_type == 2:
        midpoint = (ridge_vertices[:, :2] + ridge_vertices[:, 2:]) / 2
        radius = np.linalg.norm(midpoint - residual_points[:, :2], axis=1)
        dist = np.linalg.norm(residual_points[:, :2] - residual_points[:, 2:], axis=1)
        residual = 2 * radius * np.arcsin(dist / 2 / radius)
    elif residual_type == 3:
        residual = np.linalg.norm(residual_points[:, :2] - residual_points[:, 2:], axis=1)
    else:
        raise NotImplementedError("Only residual 1,2,3 is implemented")

    residual_w_d_potential = np.abs(boundary_potential[vor.ridge_points[:, 0]] -
                                    boundary_potential[vor.ridge_points[:, 1]])
    residual_length0 = boundary_lengths[vor.ridge_points[:, 0]]
    residual_length1 = boundary_lengths[vor.ridge_points[:, 1]]
    residual_infinit = residual_length0 != residual_length1
    residual_w = np.minimum(residual_w_d_potential, residual_length0 - residual_w_d_potential)
    if residual_type == 2:
        residual = 2 / np.pi * (np.pi / 2 * residual_w - residual)
    else:
        residual = residual_w - residual

    residual[residual_infinit] = np.inf

    # import ipdb;ipdb.set_trace()
    # residual_normalized = (residual - np.ma.masked_invalid(residual).min())/np.ptp(np.ma.masked_invalid(residual))
    return vor, residual  # _normalized


def voronoi_filter(vor, residual_normalized, tresh=4):
    indices = np.where(residual_normalized > tresh)[0]
    skel_parts = np.array(vor.ridge_vertices)[indices, :]
    include = np.where(skel_parts.min(axis=1) >= 0)[0]
    final_parts = skel_parts[include, :]
    return np.concatenate([vor.vertices[final_parts[:, 0]], vor.vertices[final_parts[:, 1]]], axis=1)


def voronoi_filter_img(img, vor, residual_normalized, tresh, return_vor=False):
    skel = voronoi_filter(vor, residual_normalized, tresh).astype(int)
    result = np.zeros_like(img)
    maxdim = [result.shape[0], result.shape[1], result.shape[0], result.shape[1]]
    for line in skel:
        if np.any(line >= maxdim) or line.min() < 0:
            continue
        rr, cc = draw.line(line[0], line[1], line[2], line[3])
        result[rr, cc] = 1
    if return_vor:
        return result * img, vor
    return result * img


def skeletonize(img, tresh=None, residual_type=3, return_vor=False):
    """
    Skeleton with voronoi skeletonization
    :param img: black background, white foreground
    :param tresh: residual threshold, if None it will be based on residual (a=1,n=3)
    :param residual_type: 1: Circularity Residual, 2: Bi-Circularity Residual, 3: Chord Residual
    :return:
    """
    if tresh is None:
        if residual_type == 2:
            tresh = 3 * (2 - 2 / np.pi * np.sqrt(2))
        else:
            tresh = 3 * (2 - np.sqrt(2))

    vor, residual = voronoi_skeletonize(img, residual_type=residual_type)
    # Our approach can end up with some non-thin parts -> send through skeletonize from skimage...
    skel = skeletonize2(voronoi_filter_img(img, vor, residual, tresh, return_vor=return_vor))
    # TODO Dirty fix: remove all small ccl
    from skimage.measure import label
    ccl = label(skel)
    unique, count = np.unique(ccl, return_counts=True)
    skel_color = np.argmax(count[1:]) + 1
    skel[(ccl != 0) & (ccl != skel_color)] = False
    return skel
