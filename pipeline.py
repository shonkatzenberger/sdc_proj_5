""" This file contains the image pipeline. """

import os
import pickle
import numpy as np
import cv2

# pylint: disable=C0103

# REVIEW shonk: For some reason, pylint can't see members of cv2, so disable 'Module <...> has no <...> member'.
# pylint: disable=E1101

# Caches the undistort function (closure).
_undistort = None

def getUndistortFunc(logger):
  """ Get a func that undistorts. """

  # pylint: disable=W0603
  global _undistort

  if _undistort is not None:
    return _undistort

  # Load the camera distortion information.
  pathPickle = os.path.abspath('./cameraCalibration.p')
  with open(pathPickle, "rb") as fh:
    vals = pickle.load(fh)
    matrixCamera = vals['matrix']
    distortionCamera = vals['distortion']
    shapeCamera = vals['shape']

  # Define and return the function.
  def _do(pixels):
    if pixels.shape[:2] != shapeCamera:
      logger.warn("Camera shape mismatch: %s vs %s", pixels.shape[:2], shapeCamera)
    else:
      pixels = cv2.undistort(pixels, matrixCamera, distortionCamera)
    return pixels

  _undistort = _do
  return _undistort

class Perspective(object):
  """ Perspective transformation class. """

  def __init__(self, dy, dx):
    assert isinstance(dy, int) and dy > 0
    assert isinstance(dx, int) and dx > 0

    # The height and width of the image.
    self._dy = dy
    self._dx = dx

    # Define the perspective transformation. These values were manually tuned so that,
    # given a standard width straight lane centered in the source image, in the destination
    # image, the lane lines are the left and right edges of self._dstPoints.
    yTop = self._dy * 0.64
    yBot = self._dy * 0.93
    xMinTop = self._dx * 0.452
    xMinBot = self._dx * 0.21
    xMinDst = self._dx * 0.25

    xLimTop = self._dx - xMinTop
    xLimBot = self._dx - xMinBot
    xLimDst = self._dx - xMinDst

    # Note that the coordinates are in (x, y) order, not the numpy standard (y, x), since that is the
    # way that OpenCv likes them.
    self._srcPts = np.float32([[xMinTop, yTop], [xLimTop, yTop], [xLimBot, yBot], [xMinBot, yBot]])
    # Note that there is really no reason that the destination shape match the source shape, but
    # doing so simplifies the canvas and image-processing code.
    self._dstPts = np.float32([[xMinDst, 0], [xLimDst, 0], [xLimDst, self._dy], [xMinDst, self._dy]])

    # Compute the perspective matrix and its inverse.
    self._matrix = cv2.getPerspectiveTransform(self._srcPts, self._dstPts)
    self._matrixInv = cv2.getPerspectiveTransform(self._dstPts, self._srcPts)

  def do(self, pixels):
    """ Apply the perspective. """
    assert isinstance(pixels, np.ndarray)
    assert len(pixels.shape) == 3 and pixels.shape[2] == 3
    return cv2.warpPerspective(pixels, self._matrix, (self._dx, self._dy), flags=cv2.INTER_LINEAR)

  def undo(self, pixels):
    """ Apply the inverse perspective. """
    assert isinstance(pixels, np.ndarray)
    assert len(pixels.shape) == 3 and pixels.shape[2] == 3
    return cv2.warpPerspective(pixels, self._matrixInv, (self._dx, self._dy), flags=cv2.INTER_LINEAR)

  # REVIEW shonk: Ideally, the return results from these properties would be immutable, but the only good
  # way to do that in python is to either clone or use tuples, both of which would be a waste of time for
  # this project.

  @property
  def srcPoints(self):
    """ The source points in (x, y) order. """
    return self._srcPts

  @property
  def dstPoints(self):
    """ The destination points in (x, y) order. """
    return self._dstPts

  @property
  def dxLane(self):
    """ The expected width of a standard lane in destination pixels. """
    return self._dstPts[2, 0] - self._dstPts[0, 0]

# This constructs and returns our main pipeline image transform function.
# The function accepts an image and returns a single-channel float32 'image',
# as well as a LaneLineInfo object.
def getPipelineFunc(logger, shape, perspective=None, sensitive=False):
  """ Returns a function that performs the pipeline transformations, including:
  * Undistort the image (adjust for camera distortion).
  * Apply perspective.
  * Convert to appropriate color space(s). The final form uses the V channel of HSV
    and the S channel of HLS, for reasons explained in the writeup.
  * Divide each (2-channel) pixel by the max of the left and right one-sided mean pools.
    This finds local maxima.
  * Apply min thresholds to the ratios.
  * Combine the 2 channels into a composite score.
  * Apply mean pooling to smooth and favor larger blobs.
  The first 3 steps (undistort, perspective, and color transformation) use numpy and
  OpenCV, while the rest use Tensorflow.

  The returned function produces a 2-D float32 result with the same spatial dimensions
  as the input. It also returns a LaneLineInfo object.

  Note that this has two modes - a normal mode and a sensitive mode. The former
  works well with clearly marked lanes in good lighting. The latter performs better
  when the lane markings aren't as clear. Note also, that there a lots of tunable
  constants inline, which could/should be lifted to parameters in real production code.
  For this project, I kept the constant tuning localized to this function.

  Note that this uses Tensorflow, so can take quite a while to construct the function.
  """

  # The input should have 3 channels.
  assert len(shape) == 3 and shape[2] == 3

  # Get the undistort function.
  undistort = getUndistortFunc(logger)

  # Get the perspective mapper, it it wasn't supplied.
  if perspective is None:
    perspective = Perspective(shape[0], shape[1])

  ###### Tunable constants.

  # The main operation is dividing each pixel by the max of its left and right mean-pool values.
  # These constants define the size of the one-sided pools. Note that the goal is to find maxes in
  # the x-direction, and that the y direction is more compressed, which is why the width of the pool
  # is so much larger than the height. Note also, that these constant definitions are a bit sloppy -
  # it would be better to define them as a fraction of 'shape'.
  dyPool = 11
  dxPool = 71

  # When dividing a pixel by the max of the one-sided pool values, we also max the denominator with
  # this. This avoids dividing by zero, but more importantly, reduces the result when a small pixel
  # is surrounded by a sea of even smaller pixels. If the pool size changes, this should also change.
  aveMin = 20

  # I tried several different variations.
  # * The first uses the L and S components of HLS.
  # * The second uses the S and V components of HSV. I found that V of HSV tends to highlight both
  #   white and yellow lines better than L of HLS. Also, the S component ended up highlighting
  #   dark tar marks, so it is effectively disabled by setting its min threshold to a huge value.
  # * The third (combo = True) uses the V component of HSV and the S component of HLS. The latter
  #   can pick up yellow lines on a bright background, like on the concrete bridges in the main
  #   project video.

  # Devine the min thresholds for the quotients. Quotient values below these are set to zero.
  useHLS = False
  combo = True
  if useHLS:
    # Use the L (0) and S (1) components of HLS.
    min0 = 1.2
    min1 = 1.3
  elif not combo:
    # From HSV, use S (0) and V (1).
    # REVIEW shonk: Don't bother processing S if we're not going to use it.
    # Currently we're just using V (by setting min0 to a high value).
    min0 = 20.0
    min1 = 1.1
  else:
    # From HSV, use V (0), and from HLS use S (1).
    # Normal mode uses 1.2, while sensitive mode uses 1.1
    min0 = 1.1 if sensitive else 1.2
    min1 = 1.3

  # Final pooling window size.
  dyWin = 20
  dxWin = 30

  # The threshold for the 2nd mask. Set to zero to disable.
  # Normal mode uses 1.0, while sensitive mode uses 0.3.
  maskThresh = 0.3 if sensitive else 1.0

  ###### End tunable constants.

  # This uses tensorflow. We import it here, rather than at the top of this file, so we don't always
  # pay the initialization hit when launching gui_show.py or other modules that might happen to
  # reference this module. This way, the price is paid only when needed.
  import tensorflow as _tf

  # Create the placeholder and cast to float32. Note that we drop the first (H) component, so
  # the input only has two channels.
  shapeInp = (1,) + shape[:2] + (2,)
  ph = _tf.placeholder(dtype=np.uint8, shape=shapeInp)
  src = _tf.to_float(ph)

  # Average pool in the y-direction with padding, then in the x-direction without padding.
  pool1 = _tf.nn.pool(src, window_shape=(dyPool, 1), pooling_type='AVG', padding='SAME')
  pool2 = _tf.nn.pool(pool1, window_shape=(1, dxPool), pooling_type='AVG', padding='VALID')

  # Pad the results in the x-direction, using symmetric for the heck of it. Pad on opposite sides for
  # the two one-sided pools. Note that this doesn't really compute proper values near the left and right
  # edges, but is a reasonable approximation. Doing the precise thing would be quite a bit more complicated,
  # namely, apply convolution with ones on one side and zeros on the other. To compute the denominators would
  # require either special construction of the tensor, or applying the same convolution kernels to an all-ones
  # input. The extra complexity probably isn't worth it. In a real production scenario, all this could/should be
  # hand-crafted CUDA (or OpenCL) code.
  poolLeft = _tf.pad(pool2, paddings=_tf.constant(((0, 0), (0, 0), (dxPool - 1, 0), (0, 0))), mode="SYMMETRIC")
  poolRight = _tf.pad(pool2, paddings=_tf.constant(((0, 0), (0, 0), (0, dxPool - 1), (0, 0))), mode="SYMMETRIC")

  # Divide the pixel by the max of the one-sided means (and aveMin).
  maxMean = _tf.maximum(_tf.maximum(poolLeft, poolRight), aveMin)
  vals = _tf.divide(src, maxMean)

  # Apply min thresholds, and then add the components.
  mask = _tf.greater_equal(vals, _tf.constant((min0, min1), dtype=np.float32))
  mask = _tf.to_float(mask)
  vals = _tf.multiply(vals, mask)
  # We add an extra "bonus point" if both values exceed the min, hence the reduce_prod.
  vals = _tf.reduce_sum(vals, axis=3, keep_dims=True) + _tf.reduce_prod(mask, axis=3, keep_dims=True)

  # Apply the final mean pool.
  vals = _tf.nn.pool(vals, window_shape=(dyWin, dxWin), pooling_type='AVG', padding='SAME')

  # Optional 2nd mask.
  if maskThresh > 0:
    # This tends to toss small isolated high spots.
    mask = _tf.greater_equal(vals, maskThresh)
    mask = _tf.to_float(mask)
    vals = _tf.multiply(vals, mask)

  # Construct the LaneLineInfo object. We use the same object repeatedly, since it carries state
  # from one frame to the next.
  lineInfo = LaneLineInfo(shape[:2], dxLane=perspective.dxLane)

  # Define and return the function. Of course, we need a TF Session object.
  sess = _tf.Session()

  def _do(pixels):
    assert pixels.shape == shape
    assert pixels.dtype == np.uint8

    # Undistort
    pixels = undistort(pixels)

    # Apply perspective
    pixels = perspective.do(pixels)

    # Color space conversion.
    if useHLS:
      hxxVals = cv2.cvtColor(pixels, cv2.COLOR_RGB2HLS)
    elif not combo:
      hxxVals = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    else:
      # REVIEW shonk: This could be done in TF to avoid the extra work of computing channels that
      # we don't use.
      hxxVals = cv2.cvtColor(pixels, cv2.COLOR_RGB2HLS)
      hsvVals = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
      hxxVals[:, :, 1] = hsvVals[:, :, 2]

    # Drop the H component when sending to tensor flow.
    fd = {ph: np.reshape(hxxVals[:, :, 1:], newshape=shapeInp)}

    # Run the TF graph and reshape to drop the batch and channel dimensions (both 1).
    res = sess.run((vals,), feed_dict=fd)
    assert len(res) == 1
    res = np.reshape(res[0], newshape=pixels.shape[:2])

    # Update the lane line info object.
    lineInfo.update(res)

    return res, lineInfo

  return _do

class LaneLineInfo(object):
  """ Tracks lane line information. """

  def __init__(self, shape, dxLane):
    assert isinstance(shape, tuple)
    assert len(shape) == 2
    assert dxLane > 0

    # The input shape (2D).
    self._shape = shape

    # Initialize the lane width seed and max and min lane width values. These are constant.
    self._dxLaneSeed = dxLane
    # Allow skinny lanes (for the challenge video).
    self._dxLaneMin = 0.75 * self._dxLaneSeed
    self._dxLaneMax = 1.1 * self._dxLaneSeed

    # Lane width for inferring initial point from the other line (when only one is found
    # in the first slice). This is updated from frame to frame.
    self._dxLane = None

    # The polygons and fit points for the two lines. Note that we reverse the y-coordinates,
    # so y=0 is at the bottom of the image (see the update method).
    self._coefs0 = None
    self._coefs1 = None
    self._pts0 = None
    self._pts1 = None

    # Initialize the search locations and margin. These fields also track information from
    # one frame to the next.
    self._dxMarginInit = self._dxLaneSeed // 2
    self._x0Init = self._shape[1] // 2 - self._dxLaneSeed // 2
    self._x1Init = self._x0Init + self._dxLaneSeed

    # Meters per pixel (scale) in each direction. These are used for curvature computations
    # and are constant. The x-scale is also used for the offset computation.
    # 3.7 meters for the standard lane width.
    self._scx = 3.7 / self._dxLaneSeed
    # The image is about 30 meters vertically.
    self._scy = 30 / self._shape[0]

  @property
  def coefsLeft(self):
    """ Return the left polynomial coefficients, if any. """
    return self._coefs0

  @property
  def coefsRight(self):
    """ Return the right polynomial coefficients, if any. """
    return self._coefs1

  @property
  def ptsLeft(self):
    """ Return the list of left fit points. """
    return self._pts0

  @property
  def ptsRight(self):
    """ Return the list of right fit points. """
    return self._pts1

  @property
  def curvatureLeft(self):
    """ Returns the (signed) curvature of the left curve, or None.
    This is the reciprocal of the radius. Negative means curving left.
    """
    return self._curvature(self._coefs0)

  @property
  def curvatureRight(self):
    """ Returns the (signed) curvature of the right curve, or None.
    This is the reciprocal of the radius. Negative means curving left.
    """
    return self._curvature(self._coefs1)

  @property
  def curvatureMean(self):
    """ Returns the (signed) curvature of the center of the lane, or None.
    This is the reciprocal of the radius. Negative means curving left.
    """
    if self._coefs0 is None:
      return self._curvature(self._coefs1)
    if self._coefs1 is None:
      return self._curvature(self._coefs0)

    # Average the two polynomials, then compute the curvature. This is more sound than averaging
    # the curvature values (or their reciprocals).
    coefs = tuple((u + v) / 2.0 for u, v in zip(self._coefs0, self._coefs1))
    return self._curvature(coefs)

  def _curvature(self, coefs):
    """ Returns the signed mathematical curvature. The radius of curvature is the
    absolute value of the reciprocal of this.
    """
    if coefs is None:
      return None

    assert len(coefs) == 3

    # Convert from pixels to meters.
    a = coefs[0] * self._scx / (self._scy * self._scy)
    b = coefs[1] * self._scx / self._scy

    # Note that we use a y-coordinate of zero for the bottom of the image,
    # so we're evaluating derivatives at y = 0.
    return (2 * a) / ((1 + b * b) ** (1.5))

  @property
  def offset(self):
    """ Return the offset from the center of the lane (in meters). """
    if self._coefs0 is None or self._coefs1 is None:
      return None
    # Note that we use a y-coordinate of zero for the bottom of the image,
    # so we're evaluating the polynomials at y = 0.
    d = (self._shape[1] - (self._coefs0[2] + self._coefs1[2])) / 2
    return d * self._scx

  def update(self, values):
    """ Update the line information. """

    assert isinstance(values, np.ndarray)
    assert values.shape == self._shape

    # For convenience, reverse the y-indices. This makes the bottom of the image have
    # y-coordinate zero. Note that through the magic of numpy, this is essentially "free",
    # that is, it doesn't copy data.
    values = values[::-1, :]

    # When True, we set the maxs to zero, just so we can see where they are.
    # This is for debugging / visualization.
    hiliteMaxs = False

    # Number of horizontal slices to process.
    numPart = 16

    # REVIEW shonk: Should we apply weight decay?
    # From the bottom of the image to the top, we decay by a factor of 2. It's not clear
    # whether this is really helpful, but reflects the fact that the bottom of the image
    # is more reliable than the top, since the perspective transformation stretched the
    # top significantly.
    wtDecay = 1.0 / 2.0 ** (1.0 / numPart)

    # We want at least this many points in a slice before we use it.
    minPts = 2 * self._shape[0] // numPart
    wtThresh = minPts * 0.3

    # Consider points within this fraction of the max.
    thresh = 0.95

    # For accumulating fit points and their weights. We record at most one point per slice.
    ysFit0 = []
    xsFit0 = []
    wtFit0 = []
    ysFit1 = []
    xsFit1 = []
    wtFit1 = []

    # (dy, dx) is the image shape (minus the channels).
    dy, dx = self._shape
    # When we're pretty sure of things, we use this margin for the next slice.
    # Otherwise, we use double this.
    dxMargin = dx // 20

    # Track the current fit points.
    y0 = y1 = 0
    x0 = self._x0Init
    x1 = self._x1Init
    # These are the margin values to use on each side.
    dxCur0 = self._dxMarginInit
    dxCur1 = self._dxMarginInit

    # The current weight multiplier.
    wtMul = 1.0

    # Process each slice, from the bottom up. Recally that y=0 is at the bottom of the image,
    # since we reversed the y-indices above.
    for iv in range(numPart):
      # Compute the x and y ranges.
      xMin0 = max(0, min(dx // 4, int(x0 - dxCur0)))
      xLim0 = min(dx, max(dx // 4, int(x0 + dxCur0)))
      xMin1 = max(0, min(dx - dx // 4, int(x1 - dxCur1)))
      xLim1 = min(dx, max(dx - dx // 4, int(x1 + dxCur1)))
      yMin = iv * dy // numPart
      yLim = (iv + 1) * dy // numPart

      # Alias the sub-portions of values and compute their horizontal maxes and masks.
      sub0 = values[yMin:yLim:, xMin0:xLim0]
      maxs0 = np.amax(sub0, axis=1, keepdims=True)
      mask0 = sub0 > thresh * maxs0
      sub1 = values[yMin:yLim:, xMin1:xLim1]
      maxs1 = np.amax(sub1, axis=1, keepdims=True)
      mask1 = sub1 > thresh * maxs1

      # Gather coordinates where the masks are non-zero.
      ys0, xs0 = np.where(mask0)
      ys1, xs1 = np.where(mask1)
      assert len(ys0.shape) == 1 and len(ys1.shape) == 1
      assert ys0.shape == xs0.shape and ys1.shape == xs1.shape

      # We use medians to reduce the risk of getting corrupted by outliers.
      # A more sophisticated approach would look for the largest tight cluster, favoring
      # vertical blobs over horizontal ones, and use a weighted average across the cluster.
      # REVIEW shonk: Perhaps factor the variance into the weight?

      # Initialize the weights to zero and margins to 2 * dxMargin.
      # If we record a point in this slice, we reduce the margin to dxMargin.
      w0 = 0
      w1 = 0
      dxCur0 = 2 * dxMargin
      dxCur1 = 2 * dxMargin

      if ys0.shape[0] >= minPts:
        # The left has required number of points. Compute its weight and make sure it is large enough.
        w0 = np.sum(sub0[ys0, xs0])
        if w0 <= wtThresh:
          w0 = 0
        else:
          # Compute the 'center' as the median.
          x0 = np.median(xs0) + xMin0
          y0 = np.median(ys0) + yMin
          xsFit0.append(x0)
          ysFit0.append(y0)
          wtFit0.append(w0 * wtMul)
          if iv == 0:
            # Save the initial x for the next frame.
            self._x0Init = x0
          # Use dxMargin for the next slice.
          dxCur0 = dxMargin
          if hiliteMaxs:
            sub0[mask0] = 0.0 if (iv % 2) == 0 else 0.5

      if ys1.shape[0] >= minPts:
        # The left has required number of points. Compute its weight and make sure it is large enough.
        w1 = np.sum(sub1[ys1, xs1])
        if w1 <= wtThresh:
          w1 = 0
        else:
          # Compute the 'center' as the median.
          x1 = np.median(xs1) + xMin1
          y1 = np.median(ys1) + yMin
          xsFit1.append(x1)
          ysFit1.append(y1)
          wtFit1.append(w1 * wtMul)
          if iv == 0:
            # Save the initial x for the next frame.
            self._x1Init = x1
          # Use dxMargin for the next slice.
          dxCur1 = dxMargin
          if hiliteMaxs:
            sub1[mask1] = 0.0 if (iv % 2) == 0 else 0.5

      if iv == 0:
        # For the first slice, we do special processing.
        assert 0 <= len(ysFit0) <= 1 and 0 <= len(ysFit1) <= 1
        self._dxMarginInit = dxMargin
        if w0 * w1 > 0:
          # Found both of them. Do a sanity check on distance and if all is good, update the _dxLane value accordingly.
          assert len(ysFit0) == 1 and len(ysFit1) == 1
          delta = xsFit1[0] - xsFit0[0]
          if delta < self._dxLaneMin or delta > self._dxLaneMax:
            # Uh, oh, looks fishy!
            print("Questionable initial lane width value: {}".format(delta))
            # REVIEW shonk: What should we do if self._dxLane is None?
            dxLane = self._dxLane or self._dxLaneSeed
            if w0 > w1:
              x1 = int(xsFit0[0] + dxLane)
              xsFit1[0] = x1
              ysFit1[0] = ysFit0[0]
              wtFit1[0] = wtFit0[0] / 2
            else:
              x0 = int(xsFit1[0] - dxLane)
              xsFit0[0] = x0
              ysFit0[0] = ysFit1[0]
              wtFit0[0] = wtFit1[0] / 2
          else:
            if self._dxLane is None:
              self._dxLane = delta
            else:
              momentum = 0.8
              self._dxLane = momentum * self._dxLane + (1 - momentum) * delta
        elif w0 + w1 > 0 and self._dxLane is not None:
          # We have the start of one line, but not the other, so infer the start of the other
          # using self._dxLane, but with half the weight.
          if w0 > 0:
            x1 = int(xsFit0[0] + self._dxLane)
            xsFit1.append(x1)
            ysFit1.append(ysFit0[0])
            wtFit1.append(w0 / 2)
          else:
            assert w1 > 0
            x0 = int(xsFit1[0] - self._dxLane)
            xsFit0.append(x0)
            ysFit0.append(ysFit1[0])
            wtFit0.append(w1 / 2)
        else:
          self._dxMarginInit = 2 * dxMargin

      wtMul *= wtDecay

    assert len(ysFit0) == len(xsFit0) == len(wtFit0)
    assert len(ysFit1) == len(xsFit1) == len(wtFit1)

    # We need at least three points to fit a quadratic.
    # REVIEW shonk: Should we require more? Perhaps use linear if there aren't very many points?
    if len(ysFit0) >= 3:
      coefs = np.polyfit(ysFit0, xsFit0, 2, w=wtFit0)
      self._coefs0 = coefs
      self._pts0 = (xsFit0, ysFit0)
    else:
      self._coefs0 = None
      self._pts0 = None

    if len(ysFit1) >= 3:
      coefs = np.polyfit(ysFit1, xsFit1, 2, w=wtFit1)
      self._coefs1 = coefs
      self._pts1 = (xsFit1, ysFit1)
    else:
      self._coefs1 = None
      self._pts1 = None
