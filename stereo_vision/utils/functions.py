import cv2
import sys
import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.spatial as spatial
from typing import Tuple, Dict, List

# These are type hints, they mostly make the code readable and testable
t_img = np.ndarray
t_disparity = np.ndarray
t_points = np.ndarray
t_descriptors = np.ndarray


def extract_features(img: t_img, num_features: int = 500) -> Tuple[t_points, t_descriptors]:
    """Extract keypoints and their descriptors.
    The OpenCV implementation of ORB is used as a backend.
    https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF
    fast robust local feature detector, It is based on the FAST keypoint detector and a modified version of the visual descriptor BRIEF (Binary Robust Independent Elementary Features).
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
        similarity_metric: A string with the name of the metric by which distances are calculated. It must be compatible
            with the ones that are defined for scipy.spatial.distance.cdist.

    Returns:
        A tuple of numpy arrays both sized [ N x 2]representing the similar point locations.

    """
    assert f1[0].shape[1] == f2[0].shape[1] == 2  # descriptor size
    assert f1[1].shape[1] == f2[1].shape[1] == 32  # points size

    # compute distance matrix
    distances = spatial.distance.cdist(f1[1], f2[1], metric=similarity_metric)

    # computing the indexes of src dst so that src[src_idx,:] and dst[dst,:] refer to matching points.
    # this gives us 2 lists: src_idx is just a list of descriptor indexes from img1 and dst_idx is a list of descriptor
    # indexes from img2. dst_idx contains the index that best mathces each descriptor in src_idx. 
    src_idx = np.arange(distances.shape[0])
    dst_idx = np.argmin(distances, axis=1)

    # find a boolean index of the matched pairs that is true only if a match was significant.
    # A match is considered significant if the ratio of its distance to the second best is lower than a given
    # threshold.
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


def get_max_translation(src: t_img, dst: t_img, well_aligned_thr=.1) -> int:
    """finds the maximum translation/shift between two images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        well_aligned_thr: a float representing the maximum y wise distance between valid matching points.

    Returns:
        an integer value representing the maximum translation of the camera from src to dst image
    """

    # Generate features/descriptors, filter and align them (courtesy of exercise 2)
    points_src, descriptors_src = extract_features(src)
    points_dst, descriptors_dst = extract_features(dst)

    # filter out correspondences that are not horizontally aligned using well aligned threshold
    src_points, dst_points = filter_and_align_descriptors((points_src, descriptors_src), (points_dst, descriptors_dst))

    # filter out pairs that are not well-aligned vertically (y-distance too large)
    y_diff = np.abs(src_points[:, 1] - dst_points[:, 1])
    well_aligned = y_diff < well_aligned_thr * src.shape[0]
    src_points = src_points[well_aligned]
    dst_points = dst_points[well_aligned]

    # Find the translation across the image using the descriptors and return the maximum value
    translations = src_points[:, 0] - dst_points[:, 0]  # x-axis translation
    max_translation = int(np.round(np.max(translations)))
    return max_translation


def render_disparity_hypothesis(src: t_img, dst: t_img, offset: int, pad_size: int) -> t_disparity:
    """Calculates the agreement between the shifted src image and the dst image.

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation

    Returns:
        a numpy array of shape [H x W] containing the euclidean distance between RGB values of the shifted src and dst
        images.
    """

    # Pad necessary values to src and dst
    src_padded = np.pad(src.astype(np.float64), ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    dst_padded = np.pad(dst.astype(np.float64), ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)

    # find the disparity value and return
    # For disparity d: compare src[x] with dst[x - d] (dst shifted right by d)
    output_width = dst.shape[1] + pad_size
    disparity = np.linalg.norm(
        src_padded[:, pad_size:pad_size + output_width, :] -
        dst_padded[:, pad_size - offset:pad_size - offset + output_width, :],
        axis=2)

    return disparity


def disparity_map(src: t_img, dst: t_img, offset: int, pad_size: int, sigma_x: int, sigma_z: int,
                  median_filter_size: int) -> t_disparity:
    """calculates the best/minimum disparity map for a given pair of images

    Args:
        src: one image taken from a camera, numpy array of shape [H x W x 3]
        dst: another image with camera only translate, numpy array of shape [H x W x 3]
        offset: an integer value by which the image is shifted
        pad_size: an integer value to pad the images for computation
        sigma_x: an integer value for standard deviation in x-direction for gaussian filter
        sigma_z: an integer value for standard deviation in z-direction for gaussian filter
        median_filter_size: an integer value representing the window size for applying median filter

    Returns:
        a numpy array of shape [H x W] containing the minimum/best disparity values for a pair of images
    """
    # Construct a stack of all reasonable disparity hypotheses.
    disparity_stack = np.array([render_disparity_hypothesis(src, dst, d, pad_size) for d in range(offset)])

    # Enforce the coherence between x-axis and disparity-axis using a 3D gaussian filter onto
    # the stack of disparity hypotheses
    disparity_stack_filtered = scipy.ndimage.gaussian_filter(disparity_stack, sigma=(sigma_z, sigma_x, sigma_x))

    # Choose the best disparity hypothesis for every pixel
    best_disparity = np.argmin(disparity_stack_filtered, axis=0)

    # Apply the median filter to enhance local consensus
    result = scipy.ndimage.median_filter(best_disparity, size=median_filter_size)

    return result


def bilinear_grid_sample(img: t_img, x_array: t_img, y_array: t_img) -> t_img:
    """Sample an image according to a sampling vector field.

    Args:
        img: one image, numpy array of shape [H x W x 3]
        x_array: a numpy array of [H' x W'] representing the x coordinates src x-direction
        y_array: a numpy array of [H' x W'] representing interpolation in y-direction

    Returns:
        An image of size [H' x W'] containing the sampled points in
    """

    # Estimate the left, top, right, bottom integer parts (l, r, t, b)
    # and the corresponding coefficients (a, b, 1-a, 1-b) of each pixel
    H, W, C = img.shape
    x0 = np.floor(x_array).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y_array).astype(np.int32)
    y1 = y0 + 1

    a = x_array - x0
    b = y_array - y0

    # Take care of out of image coordinates
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    # Produce a weighted sum of each rounded corner of the pixel
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (1 - a) * (1 - b)
    wb = (1 - a) * b
    wc = a * (1 - b)
    wd = a * b

    # Accumulate and return all the weighted four corners
    result = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id

    return result
