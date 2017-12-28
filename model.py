#pylint: disable=C0111
#pylint: disable=C0103

import collections
import cv2
import numpy as np
import tensorflow as _tf
import scipy.ndimage.measurements as _meas

# NOTE: For clarity I use names starting with g_ for tensorflow graph nodes.

# This is the factor by which the CNN shrinks spatial dimensions (when sampleMode is 1).
_spatialShrinkFactor = 32

# Builds the CNN model, for either training or prediction. The model has five convolutions that
# use stride 2, thus reducing the spatial dimensions by a factor of 2^5 = 32. These are followed
# by some 1x1 convolutions that essentially play the role of the fully connected layers in
# a typical DNN. This uses all convolutions so it can be applied to images larger than those
# used for training. Thus, the convolutions take care of "windowing" for us, keeping the bulk of
# the number crunching in the GPU, where it belongs.
#
# For training, the weights parameter should be None, in which case, a weights dictionary is allocated and a
# variable is introduced and added to the dictionary for each weight tensor. For prediction, the weights
# parameter is the dictionary of previously trained weights to use.
#
# The 'sampleMode' parameter controls additional sampling, useful for prediction (not used when training).
# When sampleMode=2, the last striding convolution is modified to use stride 1. This effectively adds
# additional outputs half-way (spatially) between the normal outputs.
# When sampleMode=4, the last two striding convolutions are modified to use stride 1 and the last
# striding convolution is modified to use dilation of 2. This effectively adds additional outputs quarter-ways
# (spatially) between the normal outputs.
#
# This returns a list of layers and the weights dictionary. When training, the weight dictionary values
# are tensorflow variables.
def buildModel(rand, src=None, batchSize=1, shape=(64, 64, 3), weights=None, sampleMode=1):
  # The input shape should have three dimensions, with each dimension divisible by _spatialShrinkFactor.
  assert len(shape) == 3
  assert shape[0] % _spatialShrinkFactor == 0, "Bad size: {}".format(shape)
  assert shape[1] % _spatialShrinkFactor == 0, "Bad size: {}".format(shape)

  # This currently handles only sampleMode values of 1, 2, and 4, but could easily be extended
  # to handle 8, 16, and 32 as well.
  assert sampleMode in (1, 2, 4)

  if weights is None:
    # Building the graph for training. Initialize weights as an empty ordered dictionary.
    weights = collections.OrderedDict()
    frozen = False
  else:
    # Building the graph for prediction. We shouldn't add anything to the dictionary.
    frozen = True

  # Mean and standard deviation for weight initialization (when training).
  mu = 0
  sigma = 0.1

  def _Conv(name, inp, count, size=3, stride=1, dilation=1, relu=True):
    """ Apply a convolution, with the given filter count, size, stride, and dilation. Does not pad. """

    # Fetch or create the bias weights.
    nameBias = name + '.bias'
    bias = weights.get(nameBias, None)
    if bias is None:
      assert not frozen, "Weight {} not found!".format(nameBias)
      # Initialize bias with zeros.
      bias = _tf.Variable(_tf.zeros(shape=(count,)), name=nameBias)
      weights[nameBias] = bias
    elif isinstance(bias, np.ndarray):
      bias = _tf.constant(bias, name=nameBias)

    # Fetch or create the kernel weights.
    nameKern = name + '.kern'
    kern = weights.get(nameKern, None)
    if kern is None:
      assert not frozen, "Weight {} not found!".format(nameKern)
      # Initialize kernel weights with truncated normal.
      assert rand is not None
      seed = rand.randint(0x80000000)
      kern = _tf.Variable(
        _tf.truncated_normal(
          shape=(size, size, inp.shape[-1].value, count),
          mean=mu, stddev=sigma, seed=seed),
        name=nameKern)
      weights[nameKern] = kern
    elif isinstance(kern, np.ndarray):
      kern = _tf.constant(kern, name=nameKern)

    # Note that we use 'VALID' padding mode. We do explicit padding at the pixel level (below)
    # so we're not repeatedly padding with zeros on each convolution.
    res = _tf.nn.convolution(
      input=inp, filter=kern,
      strides=(stride, stride), dilation_rate=(dilation, dilation),
      padding='VALID', name=name + '.conv')
    res = _tf.nn.bias_add(res, bias, name=name + '.add')
    if relu:
      res = _tf.nn.relu(res, name=name + '.relu')
    return res

  # The list of layers to return.
  layers = []

  # If 'src' isn't specified, create the input placeholder.
  cur = src if src is not None else _tf.placeholder(dtype=_tf.uint8, shape=(batchSize,) + shape, name='input')
  layers.append(cur)

  # Convert to float, and normalize to be in the range [-0.5, 0.5).
  cur = _tf.to_float(cur)
  cur = cur / 256.0
  cur = cur - 0.5
  layers.append(cur)

  # Now explictly pad by _spatialShrinkFactor - 1. This is better than padding with zeros
  # in each convolution. Note also that we pad with real pixel values (using 'SYMMETRIC' mode).
  assert _spatialShrinkFactor % 2 == 0
  dz = _spatialShrinkFactor // 2
  cur = _tf.pad(cur, paddings=((0, 0), (dz - 1, dz), (dz - 1, dz), (0, 0)), mode="SYMMETRIC")
  layers.append(cur)

  cur = _Conv('Conv1', cur, count=24, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv2', cur, count=36, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv3', cur, count=48, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv4', cur, count=64, size=3, stride=2 if sampleMode < 4 else 1)
  layers.append(cur)
  cur = _Conv('Conv5', cur, count=128, size=3, stride=2 if sampleMode < 2 else 1, dilation=1 if sampleMode < 4 else 2)
  layers.append(cur)
  # These last couple layers mimic fully connected layers by using 1x1 convolution.
  cur = _Conv('Conv6', cur, count=100, size=1, stride=1)
  layers.append(cur)
  # Don't apply relu at the end. Sigmoid is the final activation function and is applied by the caller.
  cur = _Conv('Conv7', cur, count=1, size=1, stride=1, relu=False)
  layers.append(cur)

  # Return the layers and the weights.
  return layers, weights

def saveModelWeights(sess, weights):
  """ Save trained model weights in 'model.npz'. """
  wts = collections.OrderedDict()
  for k, v in weights.items():
    w = sess.run(v, feed_dict={})
    assert isinstance(w, np.ndarray)
    print("Trained weight '{}' has shape: {}".format(k, w.shape))
    wts[k] = w
  np.savez('model.npz', **wts)

def getModelRectsMultiFunc(scales=(0.5, 1, 1.5, 2, 3, 4), flip=True):
  """ Create and return a function to apply the model at the indicated scales.
  If flip is True, the function applies the model to both the image and the image
  flipped horizontally.

  The returned function accepts an image and returns a list of 'raw' rectangles.
  """
  assert isinstance(scales, tuple) and len(scales) > 0

  # Load the previously trained weights.
  weights = dict(np.load('model.npz'))

  # Create the tensorflow session and use the same session for each model application.
  sess = _tf.Session()

  # Collect up the individual functions for the indicated scales.
  fns = []
  for scale in scales:
    # The parameters for getModelRectsFunc are hand tuned to be valid for the CNN and to cover the interesting
    # portion of the image. Note that we've hard-coded the image size (720, 1280) throughout. This is less
    # than ideal but making this more general would also make it trickier to implement and understand.
    #
    # Note that the final code (in gui_show.py) uses scales (0.5, 1, 1.5, 2), but not 3 and 4.
    if scale == 4:
      # 128 pixel rectangles.
      fns.append(getModelRectsFunc(
        scale=scale, weights=weights, sess=sess,
        shapeInpFull=(2 * 128, 1280), yLim=720 - 96, sampleMode=4))
    elif scale == 3:
      # 96 pixel rectangles.
      fns.append(getModelRectsFunc(
        scale=scale, weights=weights, sess=sess,
        shapeInpFull=(3 * 96, 1280 - 32), yLim=720 - 96, sampleMode=4))
    elif scale == 2:
      # 64 pixel rectangles.
      fns.append(getModelRectsFunc(
        scale=scale, weights=weights, sess=sess,
        shapeInpFull=(4 * 64, 1280), yLim=720 - 96, sampleMode=4))
    elif scale == 1.5:
      # 48 pixel rectangles.
      fns.append(getModelRectsFunc(
        scale=scale, weights=weights, sess=sess,
        shapeInpFull=(4 * 48, 1248), yLim=720 - 128 - 32, sampleMode=4))
    elif scale == 1:
      # 32 pixel rectangles. This one doesn't scale the image.
      fns.append(getModelRectsFunc(
        scale=scale, weights=weights, sess=sess,
        shapeInpFull=(5 * 32, 1280), yLim=720 - 128 - 48, sampleMode=4))
    elif scale == 0.5:
      # 16 pixel rectangles. This one actually scales the image up (rather than down).
      fns.append(getModelRectsFunc(
        scale=scale, sess=sess, weights=weights,
        shapeInpFull=(10 * 16, 800), yLim=720 - 128 - 48, sampleMode=4))
    else:
      assert False, "Unimplemented scale factor"

  def _do(src):
    assert src.shape == (720, 1280, 3)

    # Apply the model functions, collecting up the rectangles in a list.
    rects = []
    for fn in fns:
      rects.append(fn(src))
    if flip:
      # Apply the model functions to the flipped image.
      src = src[:, ::-1, :]
      for fn in fns:
        rcs = fn(src)
        # NOTE: we can't use a single np.subtract with out, since the source and destination
        # buffer are the same, but flipped. numpy doesn't protect against this.
        # np.subtract(src.shape[1], rcs[:, ::-1, 0], out=rcs[:, :, 0])
        rcs[:, :, 0] = src.shape[1] - rcs[:, ::-1, 0]
        rects.append(rcs)
    res = np.concatenate(rects, axis=0)
    return res

  # Return the function that accepts the image and returns the list of rectangles.
  return _do

def getModelRectsFunc(
    scale, weights, shapeInpFull,
    shapeSrc=(720, 1280), yLim=None, xOffset=0, sess=None, sampleMode=1
  ):
  """ Create and return a function to apply the model at the given scale. shapeInpFull is the
  shape of the portion of the image to use. This portion is assumed to be centered horizontally
  (overridable by xOffset), and have its bottom edge at yLim.
  """

  assert len(shapeInpFull) == 2
  assert len(shapeSrc) == 2

  assert sampleMode in (1, 2, 4)

  # Scale should be half of a positive integer.
  assert scale > 0
  assert int(scale * 2) == scale * 2
  # Get double the scale.
  scale2 = int(scale * 2)

  # Verify that everything is valid.
  assert 0 < shapeInpFull[0] <= shapeSrc[0]
  assert 0 < shapeInpFull[1] <= shapeSrc[1]
  assert all((2 * x) % (scale2 * _spatialShrinkFactor) == 0 for x in shapeInpFull), \
    "Mismatch between scale and shapeInpFull: {}".format(tuple(2 * x / scale2 for x in shapeInpFull))

  # Compute the shape of the model input.
  shapeInp = tuple((2 * x) // scale2 for x in shapeInpFull)

  # Compute the portion of the image to use.
  xMin = (shapeSrc[1] - shapeInpFull[1]) // 2 + xOffset
  assert 0 <= xMin
  xLim = xMin + shapeInpFull[1]
  assert xLim <= shapeSrc[1]

  if yLim is None:
    # Default is to extend to the bottom.
    yLim = shapeSrc[0]
  assert shapeInpFull[0] <= yLim <= shapeSrc[0]
  yMin = yLim - shapeInpFull[0]
  print("Scale {} has active region: [{} to {}] by [{} to {}]".format(scale, yMin, yLim, xMin, xLim))

  # Build the model for the size we need.
  layers, _ = buildModel(rand=None, src=None, batchSize=1, shape=shapeInp + (3,), weights=weights, sampleMode=sampleMode)
  print("Layer information for scale {}:".format(scale))
  for lay in layers:
    print("  {}: {}".format(lay.name, lay.get_shape()))

  # Get the input and output nodes. Note that g_ indicates that these are graph nodes.
  g_inp = layers[0]
  g_out = layers[-1]

  # Get the actual shrink factor (depending on sampleMode).
  assert _spatialShrinkFactor % sampleMode == 0
  shrinkFactor = _spatialShrinkFactor // sampleMode

  # Validate the model output shape.
  shapeOut = tuple(g_out.get_shape().as_list())
  assert shapeOut == (1, shapeInp[0] // shrinkFactor - (sampleMode - 1), shapeInp[1] // shrinkFactor - (sampleMode - 1), 1)

  # Drop the leading and trailing 1 from shapeOut.
  shapeOut = shapeOut[1:3]

  # Compute the scale factor to get back to source pixels.
  scaleToSrc = int(shrinkFactor * scale)
  assert scaleToSrc == shrinkFactor * scale
  # print("Scale to src: ", scaleToSrc)

  # Apply sigmoid and apply a threshold. Ideally the threshold would be client settable, but
  # for simplicity, I just hard code it here. g_outRes is a 2-d boolean tensor indicating
  # positive detections.
  # REVIEW shonk: What threshold should we use?
  g_outRes = _tf.greater(_tf.sigmoid(g_out), 0.70)
  g_outRes = _tf.reshape(g_outRes, shape=shapeOut)

  if sess is None:
    sess = _tf.Session()

  # This is the function that we'll return. It accepts an input image and returns
  # a numpy array of rectangles in ((xMin, yMin), (xLim, yLim)) order.
  def _do(src):
    assert src.shape[:2] == shapeSrc

    # Get the input portion of src.
    inp = src[yMin:yLim, xMin:xLim, :]

    # Resize if needed.
    assert inp.shape[:2] == shapeInpFull
    if shapeInp != shapeInpFull:
      inp = cv2.resize(inp, shapeInp[::-1])
    assert inp.shape[:2] == shapeInp

    # Convert to Lab, which is what the model was trained on.
    inp = cv2.cvtColor(inp, cv2.COLOR_RGB2Lab)

    # Invoke the model (in tensorflow).
    res = sess.run(g_outRes, feed_dict={g_inp: inp[None, :, :, :]})
    assert isinstance(res, np.ndarray)
    assert res.shape == shapeOut
    assert res.dtype == np.bool_

    # Add the rectangles.
    # Set showAll = True to see all the rectangles (used for crafting the coverage and density).
    showAll = False
    if showAll:
      res[:, :] = True

    # Get the non-zero cell indices.
    iys, ixs = res.nonzero()
    iys = iys.astype(np.int32)
    ixs = ixs.astype(np.int32)

    # Construct the rectangles.
    ysMin = scaleToSrc * iys + yMin
    xsMin = scaleToSrc * ixs + xMin
    ysLim = ysMin + scaleToSrc * sampleMode
    xsLim = xsMin + scaleToSrc * sampleMode
    rcs = np.stack((xsMin, ysMin, xsLim, ysLim), axis=1)
    rcs = np.reshape(rcs, newshape=(-1, 2, 2))

    return rcs

  return _do

class HeatMap(object):
  """ The heat map class. Tracks prediction density across frames. The number of
  frames is the length of the 'weights' tuple. The heat from a frame decays as more
  frames are processed.
  """

  # Since all rectangles have coordinates divisible by 4, we don't need to
  # maintain heat for every pixel, just for each 4x4 patch. This decreases
  # storage and speeds things up a bit. Note that we could also restrict to
  # the model coverage region, but I didn't bother doing this.
  def __init__(
      self, shape=(720, 1280), spatialQuant=4,
      weights=(10, 10, 8, 8, 6, 6, 4, 4, 2, 2)
    ):

    assert isinstance(shape, tuple) and len(shape) == 2
    assert isinstance(spatialQuant, int) and spatialQuant > 0
    assert all(z % spatialQuant == 0 for z in shape)
    assert isinstance(weights, tuple) and len(weights) > 0 and all(x > 0 for x in weights)

    # The rectangle coordinates that we generate are always divisible by this.
    # This saves some time and space.
    self._spatialQuant = spatialQuant

    self._shape = shape
    self._shapeMap = tuple(z // spatialQuant for z in shape)

    # These are the weights for the frames that we're tracking.
    self._weights = weights
    # Compute the weight deltas. This is the amount by which to adjust heat for old frames
    # when processing a new frame.
    self._deltas = tuple(w - weights[i + 1] for i, w in enumerate(weights[:-1]))
    assert len(self._deltas) == len(self._weights) - 1

    # The thresholds. First, we ignore any heat less than _threshLo. Then we ignore any blobs
    # with max heat below _threshHi.
    # REVIEW shonk: What should the thresholds be?
    self._threshLo = 10 * sum(self._weights)
    self._threshHi = 20 * sum(self._weights)

    # Ignore bounding rectangles smaller than this in either dimension.
    self._dzMin = 32

    # A queue of tuples of rectangles. These are the rectangles for the frames that are still active.
    self._rects = collections.deque()
    # Use ints so round off isn't an issue (we need associativity, which floating point doesn't have).
    self._heat = np.zeros(self._shapeMap, dtype=np.int32)

  def update(self, rects):
    """ Process the rectangles (possibly empty) for a new frame. """
    assert len(self._rects) <= len(self._weights)
    # rects should be an array of shape (N, 2, 2), where each 'row' is of the form ((xMin, yMin), (xLim, yLim)).
    assert isinstance(rects, np.ndarray) and rects.dtype == np.int32
    assert len(rects.shape) == 3 and rects.shape[1] == 2 and rects.shape[2] == 2

    # Convert to our heat map coordinates.
    rects = self._convertRects(rects)

    if len(self._rects) >= len(self._weights):
      # The queue is full, so toss the oldest one.
      rcs = self._rects.pop()
      self._adjustHeat(rcs, -self._weights[-1])

    # Adjust the heat contribution for old frames.
    for i, rcs in enumerate(self._rects):
      value = -self._deltas[i]
      if value != 0:
        self._adjustHeat(rcs, value)

    assert self._heat.min() >= 0

    # Add in the new rectangles.
    self._adjustHeat(rects, self._weights[0])
    self._rects.appendleft(rects)
    assert len(self._rects) <= len(self._weights)
    print("New Max: {}".format(self._heat.max()))

  def _convertRects(self, rects):
    """ Given initial rectangles, scale them to our heat tensor. """
    q = self._spatialQuant
    # assert np.all(rects % q == 0)
    # assert np.all(rects >= 0)
    # assert np.all(rects[:, :, 0] <= self._shape[1])
    # assert np.all(rects[:, :, 1] <= self._shape[0])
    # assert np.all(rects[:, 0, :] < rects[:, 1, :])

    res = np.floor_divide(rects, q)
    return res

  def _adjustHeat(self, rects, value):
    """ Adjust heat by the given value for the given rectangles. """
    assert isinstance(rects, np.ndarray) and rects.dtype == np.int32
    assert len(rects.shape) == 3 and rects.shape[1] == 2 and rects.shape[2] == 2

    # REVIEW shonk: Is there a way to do this in pure numpy land, without the
    # python for-loop?
    for rc in rects:
      self._heat[rc[0, 1]:rc[1, 1], rc[0, 0]:rc[1, 0]] += value

  def getBounds(self):
    """ Get current bounding rectangles. """

    # Ignore locations less than _threshLo.
    mask = self._heat >= self._threshLo

    # Find and label the blobs.
    labels, count = _meas.label(mask)

    q = self._spatialQuant
    dzMin = self._dzMin

    rects = []
    for i in range(1, count + 1):
      ys, xs = (labels == i).nonzero()
      rc = ((q * min(xs), q * min(ys)), (q * (max(xs) + 1), q * (max(ys) + 1)))
      if rc[1][0] - rc[0][0] < dzMin or rc[1][1] - rc[0][1] < dzMin:
        print("Rejected for size: {}".format(rc))
        continue

      # Ignore blobs whose max is too small.
      tmp = self._heat[labels == i]
      hi = tmp.max()
      if hi < self._threshHi:
        print("Rejected for max: {}, {}".format(rc, hi))
        continue

      # print("  Min/max for {} is {}/{}".format(rc, tmp.min(), hi))
      rects.append(rc)

    return rects

  def render(self, pixels=None):
    """ Render the current heat map into the given buffer and return it.
    """

    if pixels is None:
      pixels = np.zeros(shape=self._shape + (3,), dtype=np.uint8)
    else:
      # Clear the green and blue channels.
      assert pixels.shape == self._shape + (3,)
      pixels[:, :, 1:] = 0

    buf = self._heat.astype(np.float32)

    # First map _threshLo to 1 and _threshHi to 2.
    diff = self._threshHi - self._threshLo
    np.add(buf, diff - self._threshLo, out=buf)
    np.divide(buf, diff, out=buf)

    # Set anything below 1 to zero and anything above 2 to 2.
    buf[buf < 1] = 0
    np.minimum(buf, 2, out=buf)
    # Now multiply by 0x7F and cast to uint8 to get valid pixel values.
    np.multiply(buf, 0x7F, out=buf)
    buf = buf.astype(np.uint8)

    # Tile to adjust for spatial quantization.
    q = self._spatialQuant
    if q > 1:
      buf = buf[:, None, :, None]
      buf = np.tile(buf, (1, q, 1, q))
      buf = np.reshape(buf, newshape=self._shape)

    # Use the red channel for heat.
    pixels[:, :, 0] = buf
    return pixels
