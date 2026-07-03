import numpy
import cv2
from typing import Any, Tuple, Dict, List, TypeAlias
import numpy as np
import numpy.typing as npt
import scipy.spatial as spatial
from itertools import product
import os

# These are type hints, they mostly make the code readable and testable
t_array: TypeAlias = npt.NDArray[Any]
t_points: TypeAlias = t_array
t_descriptors: TypeAlias = t_array
t_homography: TypeAlias = t_array
t_img: TypeAlias = t_array
t_images: TypeAlias = Dict[str, t_img]
t_homographies: TypeAlias = Dict[Tuple[str, str], t_homography]  # The keys are the keys of src and destination images
t_image_list: TypeAlias = List[t_img]
t_str_list: TypeAlias = List[str]

np.set_printoptions(edgeitems=30, linewidth=180,
                    formatter=dict(float=lambda x: "%8.05f" % x))


def show_images(images: t_image_list, names: t_str_list) -> None:
    """Shows one or more images at once.

    Displaying a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        names: A list of strings that will appear as the window titles for each image

    Returns:
        None
    """
    for image_index in range(0, len(images)):
        cv2.imshow(names[image_index], images[image_index])
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    

def save_images(images: t_image_list, filenames: t_str_list, **kwargs) -> None:
    """Saves one or more images at once.

    Saving a single image can be done by putting it in a list.

    Args:
        images: A list of numpy arrays in opencv format [HxW] or [HxWxC]
        filenames: A list of strings where each respective file will be created

    Returns:
        None
    """
    for image_index in range(0, len(images)):
        file_name = filenames[image_index]
        _create_directory(os.path.dirname(file_name))

        cv2.imwrite(file_name, images[image_index])


def extract_features(img: t_img, num_features: int = 500) -> Tuple[t_points, t_descriptors]:
    """Extracts key-points and their descriptors.
    The OpenCV implementation of ORB (Oriented FAST and Rotated BRIEF) is used as a backend.
    It is based on the FAST key-point detector and a modified version of the visual descriptor BRIEF (Binary Robust Independent Elementary Features).
    Its aim is to provide a fast and efficient alternative to SIFT.

    Args:
        img: a numpy array of [H x Wx 3] size with byte values.
        num_features: an integer signifying how many points we desire.

    Returns:
        A tuple containing a numpy array of [N x 2] and numpy array of [N x 32]
    """
    # create an ORB detector
    orb = cv2.ORB_create(nfeatures=num_features)

    # detect keypoints and compute feature descriptors
    kp, desc = orb.detectAndCompute(img, None)

    # convert cv2.KeyPoint list to [N x 2] numpy array of (x, y) coordinates
    points = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.empty((0, 2), dtype=np.float32)
    if desc is None:
        desc = np.empty((0, 32), dtype=np.uint8)

    return points, desc


def filter_and_align_descriptors(f1: Tuple[t_points, t_descriptors], f2: Tuple[t_points, t_descriptors],
                                 similarity_threshold=.7, similarity_metric='hamming') -> Tuple[t_points, t_points]:
    """Aligns pairs of keypoints from two images.
    Aligns keypoints from two images based on descriptor similarity.
    If K points have been detected in image1 and J points have been detected in image2, the result will be to sets of N
    points representing points with similar descriptors; where N <= J and K <=points.

    Args:
        f1: A tuple of two numpy arrays with the first array having dimensions [N x 2] and the second one [N x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        f2: A tuple of two numpy arrays with the first array having dimensions [J x 2] and the second one [J x M]. M
            representing the dimensionality of the point features. In the case of ORB features, M is 32.
        similarity_threshold: The ratio the distance of most similar descriptor in image2 to the distance of the second
            most similar ratio.
        similarity_metric: A string with the name of the metric by witch distances are calculated. It must be compatible
            with the ones that are defined for scipy.spatial.distance.cdist.

    Returns:
        A tuple of numpy arrays both sized [N x 2] representing the similar point locations.

    """
    assert f1[0].shape[1] == f2[0].shape[1] == 2  # descriptor size
    assert f1[1].shape[1] == f2[1].shape[1] == 32  # points size

    # compute distance matrix (1 to 8 lines)
    distances = spatial.distance.cdist(f1[1], f2[1], metric=similarity_metric)

    # computing the indexes of src dst so that src[src_idx,:] and dst[dst,:] refer to matching points.
    # this gives us 2 lists: src_idx is just a list of descriptor indexes from img1 and dst_idx is a list of descriptor
    # indexes from img2. dst_idx contains the index that best mathces each descriptor in src_idx. 
    src_idx = np.arange(distances.shape[0])
    dst_idx = np.argmin(distances, axis=1)

    # find a boolean index of the matched pairs that is true only if a match was significant.
    # A match is considered significant if the ratio of its distance to the second best is lower than a given
    # threshold.
    # Hint: use the previously computed distance matrix to find the second best match.
    # sort distances
    distances_sorted = np.sort(distances, axis=1) # in this case axis=1 means, we vary the column idx and keep row idx fixed
    
    # calc the ratio
    ratio = distances_sorted[:, 0] / distances_sorted[:, 1] # best / second best

    # compute bin mask of significant matches
    significant_matches = ratio < similarity_threshold

    # filter out non-significant matches and return the aligned points (their location only!)
    src_idx = src_idx[significant_matches]
    dst_idx = dst_idx[significant_matches]

    # f[0] contains N points of (x, y) coordinates --> return the aligned ones
    return f1[0][src_idx, :], f2[0][dst_idx, :]


def compute_homography(f1: t_points, f2: t_points) -> t_homography:
    """Computes the homography matrix given matching points.

    In order to define a homography a minimum of 4 points are needed but the homography can also be overdefined with 5
    or more points.

    Args:
        f1: A numpy array of size [N x 2] containing x and y coordinates of the source points.
        f2: A numpy array of size [N x 2] containing x and y coordinates of the destination points.

    Returns:
        A [3 x 3] numpy array containing normalised homography matrix.
    """
    # Homogeneous coordinates
    assert f1.shape[0] == f2.shape[0] >= 4

    # - Construct the (>=8) x 9 matrix A.
    A = np.zeros((f1.shape[0] * 2, 9)) # dimensions of A are 2N x 9, where N is the number of points

    # now we need to populate A
    # - For each point pair (x, y) in f1 and (x', y') in f2, we will add two rows to A.
    for i in range(f1.shape[0]):
        x, y = f1[i]
        u, v = f2[i]
        A[i*2] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
        A[i*2+1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]

    # solve with SVD
    _, _, Vt = np.linalg.svd(A)

    h = Vt[-1, :] # last row of Vt corresponds to the smallest singular value
    H = h.reshape((3, 3))
    
    # normalize so that H[2, 2] = 1
    H = H / H[2, 2]

    return H


def _get_inlier_count(src_points: t_points, dst_points: t_points, homography: t_homography,
                      distance_threshold: float) -> int:
    """Computes the number of inliers for a homography given aligned points.
    ## - Project the image points from image 1 to image 2
    ## - A point is an inlier if the distance between the projected point and
    ##      the point in image 2 is smaller than threshold.
    Args:
        src_points: a numpy array of [N x 2] containing source points.
        dst_points: a numpy array of [N x 2] containing source points.
        homography: a [3 x 3] numpy array.
        distance_threshold: a float representing the norm of the difference between to points so that they will be
            considered the same (near enough).

    Returns:
        An integer counting how many transformed source points matched destination.
    """
    assert src_points.shape[1] == dst_points.shape[1] == 2
    assert src_points.shape[0] == dst_points.shape[0]

    # create normalized coordinates for points (maybe [x, y] --> [x, y, 1]) (4 lines)
    # src_points (N, 2) --> src_points_h (N, 3)
    ones = np.ones((src_points.shape[0], 1))
    src_points_h = np.hstack((src_points, ones)) # (N, 3)

    # project the image points from image 1 to image 2 using the homography (1 line)
    projected_h = src_points_h @ homography.T # (N, 3)

    # re-normalize the projected points ([x, y, l] --> [x/l, y/l]) (1 line)
    # reshape(-1, 1): no matter how many rows (-1) we want one column (1) in the output
    norm = projected_h[:, 2].reshape(-1, 1) # (N,) reshape to (N, 1)
    projected = projected_h[:, :2] / norm # (N, 2)

    # compute and return number of inliers (3 lines)
    distances = np.linalg.norm(projected - dst_points, axis=1) # (N,)
    inliers = distances < distance_threshold # (N,)
    return int(np.sum(inliers)) # count True values in inliers


def ransac(src_features: Tuple[t_points, t_descriptors], dst_features: Tuple[t_points, t_descriptors], steps,
           distance_threshold, n_points=4, similarity_threshold=.7) -> t_homography:
    """Computes the best homography given noisy point descriptors.

    https://en.wikipedia.org/wiki/Random_sample_consensus
    
    Args:
        src_features: A tuple with points and their descriptors detected in the source image.
        dst_features: A tuple with points and their descriptors detected in the destination image.
        steps: An integer defining how many iterations to define.
        distance_threshold: A float defining how far should to points be to be considered the same.
        n_points: The number of point pairs used to compute the homography, it must be grater than 3.
        similarity_threshold: The ratio of the most similar descriptor to the second most similar in order to consider
            that descriptors from the two images match.

    Returns:
        A numpy array containing the homography.
    """

    # filter and align descriptors (1 line)
    src_points, dst_points = filter_and_align_descriptors(src_features, dst_features, similarity_threshold)

    # initialize the optimization loop
    best_count = 0
    best_homography = np.eye(3)

    # optimization loop
    for n in range(steps):
        # select random subset of points (at least 4 points) (2 lines)
        idx = np.random.choice(src_points.shape[0], size=n_points, replace=False)
        sample_src = src_points[idx]
        sample_dst = dst_points[idx]

        # compute homography for the random points
        H = compute_homography(sample_src, sample_dst)

        # compare the current homography to the current best homography and update the best homography using
        # inlier count
        count = _get_inlier_count(src_points, dst_points, H, distance_threshold)
        if count > best_count:
            best_count = count
            best_homography = H

    print(f"After {steps:4} steps: {best_count} RANSAC points match!")
    return best_homography


def propagate_homographies(homographies: t_homographies, reference_name: str) -> t_homographies:
    """Computes homographies from every image to the reference image given a homographies between all pairs of
    consecutive images.

    This method could be loosely described as applying Dijkstra's algorithm applied to exploit the commutative
    relationship of matrix multiplication and compute homography matrices between all images and any image.

    Args:
        homographies: A dictionary where the keys are tuples with the names of each image pair and the values are
            [3 x 3] arrays containing the homographies between those images.
        reference_name: The of the image which will be the destination for all homographies.

    Returns:
        A dictionary of the same form as the input mappning all images to the reference.
    """
    initial = {k: v for k, v in homographies.items()}  # deep copy
    for k, h in list(initial.items()):
        initial[(k[1], k[0])] = np.linalg.inv(h)
    initial[(reference_name, reference_name)] = np.eye(3)  # Added the identity homography for the reference
    desired = set([(k[0], reference_name) for k in homographies.keys()])
    solved = {k: v for k, v in initial.items() if k[1] == reference_name}
    while not (set(solved.keys()) >= desired):

        new_steps = set([(i, s) for i, s in product(initial.keys(), solved.keys()) if
                     s[1] != i[0] and s[0] == i[1] and s[0] != s[1] and (i[0], s[1]) not in solved.keys()])
        # s[1] != i[0] no pair who's product leads to identity
        # s[0] == i[1] only connected pairs
        # s[0]!=s[1] no identity in the solution
        # set removes duplicates

        assert len(new_steps) > 0  # not all desired can be linked to reference
        for initial_k, solved_k in new_steps:
            new_key = initial_k[0], solved_k[1]
            solved[solved_k]
            initial[initial_k]
            solved[new_key] = np.matmul(solved[solved_k], initial[initial_k])
    return solved


def compute_panorama_borders(images: t_images, homographies: t_homographies) -> Tuple[float, float, float, float]:
    """Computes the bounding box of the panorama defined the images and the homographies mapping them to the reference.

    This bounding box can have non integer and even negative coordinates.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies:  A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the second.

    Returns:
        A tuple containing the bounding box [left, top, right, bottom] of the whole panorama if stiched.

    """
    homographies = {k[0]: v for k, v in homographies.items()}  # assining homographies to their source image
    assert homographies.keys() == images.keys()  # map homographies to source image only
    all_corners = []
    for name in sorted(images.keys()):
        img, homography = images[name], homographies[name]
        width, height = img.shape[0], img.shape[1]
        corners = ((0, 0), (0, width), (height, width), (height, 0))
        corners = np.array(corners, dtype='float32')
        all_corners.append(cv2.perspectiveTransform(corners[None, :, :], homography)[0, :, :])
    all_corners = np.concatenate(all_corners, axis=0)
    left, right = np.floor(all_corners[:, 0].min()), np.ceil(all_corners[:, 0].max())
    top, bottom = np.floor(all_corners[:, 1].min()), np.ceil(all_corners[:, 1].max())
    return left, top, right, bottom


def translate_homographies(homographies: t_homographies, dx: float, dy: float) -> t_homographies:
    """Applies a uniform translation to a dictionary with homographies.

    Args:
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the second.
        dx: a float representing the horizontal displacement of the translation.
        dy: a float representing the vertical displacement of the translation.

    Returns:
        a copy of the homographies dict which maps the same keys to the translated matrices.
    """
    # create a translation matrix
    T = np.eye(3)
    T[0, 2] = dx
    T[1, 2] = dy

    # apply translation matrix on every homography matrix
    return {k: T @ v for k, v in homographies.items()}


def stitch_panorama(images: t_images, homographies: t_homographies, output_size: Tuple[int, int],
                   rendering_order: List[str] = []) -> t_img:
    """Stiches images after it reprojects them with a homography.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            mapping from the first image to the reference image.
        output_size: A tuple with integers representing the witdh and height of the resulting panorama.
        rendering_order: A list containing the names of the images representing the order in witch the images will be
            overlaid. The list must contain either all images names in some permutation or be empty in which case, the
            images will be rendered in the alphanumeric order of their names.
    Returns:
        A numpy array with the panorama image.
    """
    homographies = {k[0]: v for k, v in homographies.items()}  # assining homographies to their source image
    assert homographies.keys() == images.keys()
    if rendering_order == []:
        rendering_order = sorted(images.keys())
    panorama = np.zeros([output_size[1], output_size[0], 3], dtype=np.uint8)
    for name in rendering_order:
        rgba_img = cv2.cvtColor(images[name], cv2.COLOR_RGB2RGBA)
        rgba_img[:, :, 3] = 255
        tmp = cv2.warpPerspective(rgba_img, homographies[name], output_size, cv2.INTER_LINEAR_EXACT)
        new_pixels = ((tmp[:, :, 3] == 255)[:, :, None] & (panorama == np.zeros([1, 1, 3])))
        old_pixels = 1 - new_pixels
        panorama[:, :, :] = panorama * old_pixels + tmp[:, :, :3] * new_pixels
    return panorama


def create_stitched_image(images: t_images, homographies: t_homographies, reference_name: str,
                          rendering_order: List[str] = []) -> t_img:
    """Will create a panorama by stitching the input images after reprojecting them.

    Args:
        images: A dictionary mapping image names to numpy arrays containing images.
        homographies: A dictionary mapping Tuples with pairs image names to numpy arrays representing homographies
            that can reproject the first image to be aligned with the reference image.
        reference_name: A string with the name of the image to which all other images will be aligned.
        rendering_order: A list containing the names of the images representing the order in witch the images will be
            overlaid. The list must contain either all images names in some permutation or be empty in which case, the
            images will be rendered in the alphanumeric order of their names.
    Returns:
        A numpy array with the panorama image.
    """
    #  from homographies between consecutive images we compute all homographies from any image to the reference.
    homographies = propagate_homographies(homographies, reference_name=reference_name)
    #  lets calculate the panorama size
    left, top, right, bottom = compute_panorama_borders(images, homographies)
    width = int(1 + np.ceil(right) - np.floor(left))
    height = int(1 + np.ceil(bottom) - np.floor(top))
    #  lets make the homographies translate all images inside the panorama.
    homographies = translate_homographies(homographies, -left, -top)
    return stitch_panorama(images, homographies, (width, height), rendering_order=rendering_order)


def _create_directory(dir_path: str) -> None:
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        print(f"Error: {dir_path} - {e.strerror}")
