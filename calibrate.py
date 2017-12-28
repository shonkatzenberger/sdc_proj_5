""" For calibrating a camera from a bunch of images of chess boards.
"""

import os
import glob
import pickle
import logging

import numpy as np
import cv2

# For some reason, pylint doesn't see the internals of cv2.
# pylint: disable=E1101

def _gatherPoints(logger, nx, ny, pathWild, shape):
  """ Finds chess board image points for the images indicated by 'pathWild'. Returns
  object points, image points, and the image shape.
  """
  assert isinstance(nx, int) and nx >= 2
  assert isinstance(ny, int) and ny >= 2
  assert shape is None or isinstance(shape, tuple) and len(shape) == 2

  # Prepare object grid points.
  ptsGrid = np.mgrid[:nx, :ny, :1].T.reshape(-1, 3).astype(np.float32)

  # Arrays to store object points and image points from all the images.
  ptsImg = []

  # Make a list of calibration image paths.
  paths = glob.glob(pathWild)
  assert isinstance(paths, list) and len(paths) > 0

  # For each image, find the chess board corners and add to the pt lists.
  num = 0
  for path in paths:
    assert len(ptsImg) == num
    path = os.path.abspath(path)

    # Load the image, convert to gray scale, and find the corners.
    img = cv2.imread(path)
    shapeCur = img.shape[:2]
    if shape is None:
      shape = shapeCur

    assert isinstance(shape, tuple)
    if shape != shapeCur:
      if shape[0] > shapeCur[0] or shape[1] > shapeCur[1]:
        logger.warn("Inconsistent image shapes, %s vs %s, at '%s'! Skipping this image.", shape, shapeCur, path)
        continue

      logger.warn("Inconsistent image shapes, %s vs %s, at '%s'! Cropping this image.", shape, shapeCur, path)
      dy = (shapeCur[0] - shape[0]) // 2
      dx = (shapeCur[1] - shape[1]) // 2
      img = img[dy : dy + shape[0], dx : dx + shape[1]]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    assert isinstance(ret, bool)

    if not ret:
      logger.warn("Couldn't find chess board corners in '%s'!", path)
    else:
      ptsImg.append(corners)
      num += 1
  assert len(ptsImg) == num

  if num == 0:
    raise RuntimeError("Didn't find any chess board corners!")
  assert shape is not None
  logger.info("Gathered corners from %s images with shape %s", num, shape)

  ptsObj = [ptsGrid] * num
  assert len(ptsObj) == num

  return ptsObj, ptsImg, shape

def calibrate(logger=None, nx=9, ny=6, pathWild='./camera_cal/calibration*.jpg', pathPickle='./cameraCalibration.p', shape=None):
  """ Computes and serializes camera calibration values from a bunch of chess board images.
  The values are saved as a pickle file.
  """
  if logger is None:
    logger = logging.getLogger('calibration')

  ptsObj, ptsImg, shape = _gatherPoints(logger, nx, ny, pathWild, shape)
  err, matrix, distortion, _, _ = cv2.calibrateCamera(ptsObj, ptsImg, shape[::-1], None, None)

  logger.debug("Matrix is:\n%s", matrix)
  logger.debug("Distorion is:\n%s", distortion)

  # Pickle the camera calibration results (just the matrix and distortion values).
  vals = {
    'matrix': matrix,
    'distortion': distortion,
    'shape': shape
  }
  pathPickle = os.path.abspath(pathPickle)
  with open(pathPickle, "wb") as fh:
    pickle.dump(vals, fh)
  logger.info("Saved camera calibration values to '%s'", pathPickle)
  logger.info("Calibration error: %s", err)

def _run():
  # Initialize logging to the INFO level.
  logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s %(name)s]  %(message)s')
  # Calibrate, using the defaults.
  calibrate()

if __name__ == "__main__":
  _run()
