import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

from scipy import signal
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates

import sol4_utils

DERIVATIVE_KERNEL = np.array([1, 0, -1])
DERIVATIVE_KERNEL_SIZE = (3, 3)
X_DERIVATIVE = 1
Y_DERIVATIVE = 2
DERIVATIVES_BLUR_KERNEL_SIZE = 3
K_FACTOR = 0.04
WHITE_VAL = 1
WINDOW_SIZE = 7
GAUSSIAN_PYRAMID_WRONG_SIZE = "Error: the pyramid has wrong size"
HOMOGENEOUS_THREE_COORD_ZERO = "Error: the third homogeneous coord is zero"


def get_image_derivative(im, direction):
    """
    returns image derivative according the given direction
    """
    derivative_kernel = np.zeros(DERIVATIVE_KERNEL_SIZE)
    if (direction == X_DERIVATIVE):
        derivative_kernel[1, :] = DERIVATIVE_KERNEL
    else:
        derivative_kernel[:, 1] = DERIVATIVE_KERNEL
    return signal.convolve2d(im, derivative_kernel, boundary='symm')


def calc_r_for_harris(Ix2, IxIy, IyIx, Iy2):
    """
    given derivatives matrix - returns its R calculation
    """
    determinant = (Ix2 * Iy2) - (IxIy * IyIx)
    trace = Ix2 + Iy2
    return determinant - (K_FACTOR * (np.power(trace, 2)))


def get_corners_coordinates(res_matrix):
    """
    given response matrix returns its corners coordinates as an array
    """
    cond = res_matrix == WHITE_VAL
    return np.flip(np.asarray(np.where(cond == True)), 0).T


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing a greyscale image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    Ix = get_image_derivative(im, X_DERIVATIVE)
    Iy = get_image_derivative(im, Y_DERIVATIVE)
    Ix2_blur = sol4_utils.blur_spatial((Ix * Ix), DERIVATIVES_BLUR_KERNEL_SIZE)
    Iy2_blur = sol4_utils.blur_spatial((Iy * Iy), DERIVATIVES_BLUR_KERNEL_SIZE)
    IxIy_blur = sol4_utils.blur_spatial((Ix * Iy), DERIVATIVES_BLUR_KERNEL_SIZE)
    IyIx = Iy * Ix
    R_matrix = calc_r_for_harris(Ix2_blur, IxIy_blur, IyIx, Iy2_blur)
    corners_im = non_maximum_suppression(R_matrix)
    return get_corners_coordinates(corners_im)


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


def normalize_samples_window(window, k):
    """
    given window of samples - normalize it
    """
    window_mean = np.mean(window)
    divisor = np.linalg.norm(window - window_mean)
    if (divisor != 0):
        return ((window - window_mean) / divisor)
    return np.zeros((k, k))


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    k = 1 + (2 * desc_rad) # window size: e.g radius 3 => window 7*7
    descriptor = np.zeros((pos.shape[0], k, k)) # N matrices of k*k each
    for i in range(pos.shape[0]):
        x_window_coords = np.tile(np.linspace((pos[i][0] - desc_rad), (pos[i][0] + desc_rad), k), k)
        y_window_coords = np.repeat(np.linspace((pos[i][1] - desc_rad), (pos[i][1] + desc_rad), k), k)
        patch = map_coordinates(im, [y_window_coords, x_window_coords], order=1, prefilter=False)
        patch = normalize_samples_window(patch, k).reshape(k, k)
        descriptor[i, :, :] = patch
    return descriptor


def convert_gaussian_coordinates(origin_coords, origin_level, dest_level):
    """
    given gaussian coordinates in some level - covert them to another level
    """
    return (2 ** (origin_level - dest_level)) * origin_coords


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
    """
    if (len(pyr) < 3):
        print(GAUSSIAN_PYRAMID_WRONG_SIZE)
        exit()
    corners_L0 = spread_out_corners(pyr[0], WINDOW_SIZE, WINDOW_SIZE, 3)
    corners_L2 = convert_gaussian_coordinates(corners_L0, 0, 2)
    descriptor = sample_descriptor(pyr[2], corners_L2, 3)
    return [corners_L0, descriptor]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    desc1_flat = desc1.reshape(desc1.shape[0], desc1.shape[1]*desc1.shape[2])
    desc2_flat = desc2.reshape(desc2.shape[0], desc2.shape[1]*desc2.shape[2])
    feature_mult = np.dot(desc1_flat, desc2_flat.T)
    cond_bool = np.full(feature_mult.shape, False)
    for i in range(2):
        for j in range(feature_mult.shape[0]):
            max_from_row = feature_mult[j].argsort()[-2:][::-1]
            max1_col = feature_mult[:, max_from_row[0]]
            max2_col = feature_mult[:, max_from_row[1]]
            max_from_col1 = max1_col.argsort()[-2:][::-1]
            max_from_col2 = max2_col.argsort()[-2:][::-1]
            if ((j in max_from_col1) and (feature_mult[j][max_from_row[0]] > min_score)):
                cond_bool[j][max_from_row[0]] = True
            if ((j in max_from_col2) and (feature_mult[j][max_from_row[1]] > min_score)):
                cond_bool[j][max_from_row[1]] = True
        cond_bool = cond_bool.T
        feature_mult = feature_mult.T
    final_result = np.where(cond_bool)
    return [np.asarray(final_result[0], dtype=np.int),
            np.asarray(final_result[1], dtype=np.int)]


def create_homogeneous_coords(standard_coords_point):
    """
    given a point with standard coordinates - return its homogeneous coords
    """
    return np.array([standard_coords_point[0],
                     standard_coords_point[1],
                     1])


def create_standard_coords(standard_coords_point):
    """
    given a point with homogeneous coordinates - return its standard coords normalized
    """
    if (standard_coords_point[2] == 0):
        print(HOMOGENEOUS_THREE_COORD_ZERO)
        exit()
    return np.array([standard_coords_point[0] / standard_coords_point[2],
                     standard_coords_point[1] / standard_coords_point[2]])


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    point_coords = np.insert(pos1, 2, 1, axis=1)
    homography_applied = np.dot(H12, point_coords.T)
    return (homography_applied[:2, :] / homography_applied[2, :]).T


def get_random_indexes(pts_arr, want_two):
    """
    given array of points - return random indexes of points which are all different
    """
    randP1=np.random.randint(0, pts_arr.shape[0])
    if (want_two):
        randP2=np.random.randint(0, pts_arr.shape[0])
        while(randP2 == randP1):
            randP2=np.random.randint(0, pts_arr.shape[0])
        return np.array([randP1, randP2])
    return np.array([randP1])


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    final_set = np.zeros((0, ))
    for i in range(num_iter):
        if(not translation_only):
            p_random = get_random_indexes(points1, True)
        else:
            p_random = get_random_indexes(points1, False)
        rigid_matrix = estimate_rigid_transform(points1[p_random],
                                                points2[p_random], translation_only)
        if (rigid_matrix.any() == None):
            continue
        homography_pts = apply_homography(points1, rigid_matrix)
        norm_dist_E = np.power(np.linalg.norm(homography_pts - points2, axis=1), 2)
        inliers = np.where(norm_dist_E < inlier_tol)[0]
        if(inliers.shape[0] > final_set.shape[0]):
            final_set = inliers
    rigid_matrix = estimate_rigid_transform(points1[final_set],
                                            points2[final_set], translation_only)
    return [rigid_matrix, final_set]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    horizontal_im = np.hstack((im1, im2))
    points2[:, 0] += im1.shape[1]
    plt.imshow(horizontal_im, cmap=plt.get_cmap('gray'))
    for i in range(points1.shape[0]):
        x = [points1[i][0], points2[i][0]]
        y = [points1[i][1], points2[i][1]]
        color = 'y' if i in inliers else 'b' # outliers - blue, inliers - yellow
        plt.plot(x, y, mfc='r', c=color, lw=0.2, ms=10, marker='.')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    res = [0] * (len(H_succesive) + 1)
    res[m] = np.eye(3)
    for i in range(m):
        tmp_idx = (m - 1) - i
        res[tmp_idx] = np.dot(res[tmp_idx + 1], H_succesive[tmp_idx])
        res[tmp_idx] /= res[tmp_idx][2][2]
    for i in range(m + 1, len(res)):
        res[i] = np.dot(res[i - 1], np.linalg.inv(H_succesive[i - 1]))
        res[i] /= res[i][2][2]
    return res


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
    and the second row is the [x,y] of the bottom right corner
    """
    corners_idx = np.array(([0, 0], [0, h], [w, 0], [w, h]))
    new_corners_idx = apply_homography(corners_idx, homography)
    return np.array([[np.min(new_corners_idx[:, 0]), np.min(new_corners_idx[:, 1])],
                    [np.max(new_corners_idx[:, 0]), np.max(new_corners_idx[:, 1])]], dtype=np.int)


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    bounding_box = compute_bounding_box(homography, image.shape[0], image.shape[1])
    x_range = np.arange(bounding_box[0][0], bounding_box[1][0])
    y_range = np.arange(bounding_box[0][1], bounding_box[1][1])
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    inverse_homography = np.linalg.inv(homography)
    calc_homography = apply_homography(np.array([x_grid.flatten(), y_grid.flatten()]).T,
                                       inverse_homography).T
    return map_coordinates(image, calc_homography, order=1, prefilter=False)\
                           .reshape(bounding_box[1][1] - bounding_box[0][1],
                                    bounding_box[1][0] - bounding_box[0][0]).T


def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]

  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      plt.imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))

  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()

