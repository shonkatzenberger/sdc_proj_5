""" This contains image processing transforms. """

import numpy as np
import cv2
import skimage.feature as _feat

# pylint: disable=missing-docstring

# REVIEW shonk: For some reason, pylint has trouble finding things in cv2.
# pylint: disable=no-member

_eps = 1e-5

def standardize(image):
  """ This standardizes the image to have three channels and dtype=uint8. """
  assert 2 <= len(image.shape) <= 3

  if image.dtype != np.uint8:
    # Handle floating point dtype. In this case, take the absolute value and scale to [0, 255].
    assert image.dtype == np.float32 or image.dtype == np.float64
    image = np.absolute(image)
    # Compute the max and mean. Divide by the smaller of the max and 3 times the mean.
    den = image.max()
    avg = image.mean()
    den = min(den, 10 * avg)
    image = np.uint8(np.minimum(image / max(_eps, den), 1.0) * 255.0)
  assert image.dtype == np.uint8

  if len(image.shape) == 2:
    assert len(image.shape) == 2
    image = stack(image)

  assert len(image.shape) == 3 and image.shape[2] == 3
  return image

def stack(image):
  """ Stack a monochrome image into a 3-channel image. """
  assert len(image.shape) == 2
  return np.stack((image, image, image), axis=2)

def _channel(image, index):
  """ Extract the indicated channel. """
  assert len(image.shape) == 3
  assert 0 <= index < image.shape[2]
  return image[:, :, index]

def none(image):
  """ Do nothing. """
  return image

# RGB transforms.
def red(image):
  return _channel(image, 0)

def green(image):
  return _channel(image, 1)

def blue(image):
  return _channel(image, 2)

# Gray scale.
def gray(image):
  if len(image.shape) == 2:
    return image
  assert len(image.shape) == 3 and image.shape[2] == 3
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# HSV transforms.
def hsv(image):
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  return res

def hsv_sv(image):
  """ Returns HSV with the H component zeroed out. """
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  # Zero out the H component.
  res[:, :, 0] = 0
  return res

def hsv_vs(image):
  """ Returns HSV with the H component zeroed out. """
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  # Put the channels in 0VS order.
  res[:, :, 0] = 0
  res = np.stack((res[:, :, 0], res[:, :, 2], res[:, :, 1]), axis=2)
  print(res.shape)
  return res

def hsv_sv_min(image):
  """ Returns the min of the S and V components of HSV. """
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  res = np.minimum(res[:, :, 1], res[:, :, 2])
  return res

def hsv_sv_avg(image):
  """ Returns the average of the S and V components of HSV. """
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  res = res[:, :, 1] / 2 + res[:, :, 2] / 2
  return res

def hsv_sv_mask(image):
  """ Returns S and masked by V >= thresh. """
  assert len(image.shape) == 3 and image.shape[2] == 3
  hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  res = hsv[:, :, 1]
  res[hsv[:, :, 2] < 0x80] = 0
  return res

def hsv_h(image):
  return _channel(hsv(image), 0)

def hsv_s(image):
  return _channel(hsv(image), 1)

def hsv_v(image):
  return _channel(hsv(image), 2)

# HLS transforms.
def hls(image):
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  return res

def hls_ls(image):
  """ Returns HLS with the H component zeroed out. """
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  # Zero out the H component.
  res[:, :, 0] = 0
  return res

def hls_h(image):
  return _channel(hls(image), 0)

def hls_l(image):
  return _channel(hls(image), 1)

def hls_s(image):
  return _channel(hls(image), 2)

# Lab transforms.
def lab(image):
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
  return res

def lab_l(image):
  return _channel(lab(image), 0)

def lab_a(image):
  return _channel(lab(image), 1)

def lab_b(image):
  return _channel(lab(image), 2)

# Luv transforms.
def luv(image):
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
  return res

def luv_l(image):
  return _channel(luv(image), 0)

def luv_u(image):
  return _channel(luv(image), 1)

def luv_v(image):
  return _channel(luv(image), 2)

# YCrCb transforms.
def ycrcb(image):
  assert len(image.shape) == 3 and image.shape[2] == 3
  res = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
  return res

def ycrcb_y(image):
  return _channel(ycrcb(image), 0)

def ycrcb_cr(image):
  return _channel(ycrcb(image), 1)

def ycrcb_cb(image):
  return _channel(ycrcb(image), 2)

# Sobel derivative transforms.
def sobel_x(image):
  return _sobel(image, x=1)

def sobel_y(image):
  return _sobel(image, x=0)

def _sobel(image, x=1, ksize=9):
  assert x == 0 or x == 1
  if len(image.shape) == 3:
    assert image.shape[2] == 3
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  assert len(image.shape) == 2
  res = cv2.Sobel(image, cv2.CV_64F, x, 1 - x, dst=None, ksize=ksize)
  return np.absolute(res)

def sobel_norm(image):
  if len(image.shape) == 3:
    assert image.shape[2] == 3
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  assert len(image.shape) == 2
  x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
  y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
  return np.sqrt(np.square(x) + np.square(y))

# Laplacian.
def laplace(image, ksize=9):
  if len(image.shape) == 3:
    assert image.shape[2] == 3
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  assert len(image.shape) == 2
  lap = cv2.Laplacian(image, cv2.CV_32F, dst=None, ksize=ksize)
  np.maximum(lap, 0, out=lap)
  return lap

def hog8(image):
  assert 2 <= len(image.shape) <= 3
  if len(image.shape) == 3:
    assert image.shape[2] == 3
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  _, res = _feat.hog(
    image,
    orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
    block_norm='L2-Hys', visualise=True, transform_sqrt=True, feature_vector=False, normalise=None)
  return res

def hog16(image):
  assert 2 <= len(image.shape) <= 3
  if len(image.shape) == 3:
    assert image.shape[2] == 3
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  _, res = _feat.hog(
    image,
    orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
    block_norm='L2-Hys', visualise=True, transform_sqrt=True, feature_vector=False, normalise=None)
  return res

# An initial combination.
def combo(image, ksize=9):
  """ Used for experimenting with various combinations.
  """
  assert len(image.shape) == 3 and image.shape[2] == 3
  image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  l = image[:, :, 1]
  s = image[:, :, 2]

  mask0 = np.zeros_like(l)
  # mask0[l >= 120] = 0xFF

  sobelx = cv2.Sobel(l, cv2.CV_32F, 1, 0, dst=None, ksize=ksize)
  sobelx = np.absolute(sobelx)
  maxS = sobelx.max()
  minThresh = 0.15 * maxS
  maxThresh = 0.50 * maxS

  mask1 = np.zeros_like(l)
  mask1[(minThresh < sobelx) & (sobelx <= maxThresh)] = 0xFF

  mask2 = np.zeros_like(l)
  # mask2[(s >= 70)] = 0xFF
  mask2[(s >= 80) & (l >= 30)] = 0xFF

  return np.stack((mask0, mask1, mask2), axis=2)

# Thresholding.
def abovePercent(image, thresh=50):
  """ Keep values above the given percentage of the max value. """
  assert len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 3
  assert 0 <= thresh <= 100

  m = np.max(image, axis=(0, 1))
  res = np.copy(image)
  res[image <= (thresh / 100.0) * m] = 0
  return res

def abovePercentH(image, thresh=50):
  """ Keep values above the given percentage of the max value along each horizontal row. """
  assert len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 3
  assert 0 <= thresh <= 100

  m = np.max(image, axis=1, out=None, keepdims=True)
  res = np.copy(image)
  res[image <= (thresh / 100.0) * m] = 0
  return res

def aboveAbs(image, thresh=1.5):
  """ Keep values above an 'absolute' (as opposed to relative) value. """
  assert len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 3
  assert thresh >= 0

  res = np.copy(image)
  res[image <= thresh] = 0
  return res

def getComboPooler(shape, dtype, dy=11, dx=71, eps=20):
  """ Returns a function that divides each pixel by the max of its two one-sided mean pools. """
  assert 2 <= len(shape) <= 3

  # Uses tensorflow.
  import tensorflow as _tf

  # Create the placeholder.
  shapeInp = (1,) + shape + (() if len(shape) == 3 else (1,))
  ph = _tf.placeholder(dtype=dtype, shape=shapeInp)

  # Cast to float32 if needed.
  val = ph
  if dtype != np.float32:
    val = _tf.to_float(val)

  # Average pool in the y - direction with padding.
  pool1 = _tf.nn.pool(val, window_shape=(dy, 1), pooling_type='AVG', padding='SAME')

  # Average pool in the x - direction without padding.
  pool2 = _tf.nn.pool(pool1, window_shape=(1, dx), pooling_type='AVG', padding='VALID')

  # Pad the results, using symmetric for the heck of it.
  poolLeft = _tf.pad(pool2, paddings=_tf.constant(((0, 0), (0, 0), (dx - 1, 0), (0, 0))), mode="SYMMETRIC")
  poolRight = _tf.pad(pool2, paddings=_tf.constant(((0, 0), (0, 0), (0, dx - 1), (0, 0))), mode="SYMMETRIC")

  # Divide the pixel by the max of the one-sided means, and clip at 4.0.
  quo = _tf.minimum(_tf.divide(val, _tf.maximum(_tf.maximum(poolLeft, poolRight), eps)), 4.0)

  # Define and return the function.
  sess = _tf.Session()
  def _do(inp):
    fd = {ph: np.reshape(inp, newshape=shapeInp)}
    res = sess.run((quo,), feed_dict=fd)
    assert len(res) == 1
    return np.reshape(res[0], newshape=inp.shape)

  return _do

# A combination intended to be applied after HLS-LS and combo pooler with no threshold.
def comboAfterPool(image, ksize=9):
  assert len(image.shape) == 3 and image.shape[2] == 3

  l = image[:, :, 1]
  s = image[:, :, 2]

  mask = np.zeros(shape=image.shape, dtype=np.uint8)

  # sobelx = cv2.Sobel(l, cv2.CV_32F, 1, 0, dst=None, ksize=ksize)
  # sobelx = np.absolute(sobelx)
  # maxS = sobelx.max()
  # minThresh = 0.15 * maxS
  # mask[minThresh < sobelx, 0] = 0xFF

  mask[l >= 1.2, 1] = 0xFF
  mask[s >= 1.3, 2] = 0xFF

  return mask
