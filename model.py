#pylint: disable=C0111
#pylint: disable=C0103

import collections
import cv2
import numpy as np
import tensorflow as _tf

_factor1 = 16
def _buildModel1(rand, src=None, batchSize=1, shape=(64, 64, 3), weights=None):
  assert len(shape) == 3
  assert shape[0] % _factor1 == 0, "Bad size: {}".format(shape)
  assert shape[1] % _factor1 == 0, "Bad size: {}".format(shape)
  assert shape[2] == 3

  if weights is None:
    weights = collections.OrderedDict()
    frozen = False
  else:
    frozen = True

  # Mean and standard deviation for weight initialization.
  mu = 0
  sigma = 0.1

  def _Conv(name, inp, count, size=3, stride=1, relu=True):
    """ Apply a convolution, with the given filter count and size. Uses padding. """
    nameBias = name + '.bias'
    bias = weights.get(nameBias, None)
    if bias is None:
      assert not frozen
      bias = _tf.Variable(_tf.zeros(shape=(count,)), name=nameBias)
      weights[nameBias] = bias
    elif isinstance(bias, np.ndarray):
      bias = _tf.constant(bias, name=nameBias)

    nameKern = name + '.kern'
    kern = weights.get(nameKern, None)
    if kern is None:
      assert not frozen
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

    res = _tf.nn.convolution(input=inp, filter=kern, strides=(stride, stride), padding='SAME', name=name + '.conv')
    res = _tf.nn.bias_add(res, bias, name=name + '.add')
    if relu:
      res = _tf.nn.relu(res, name=name + '.relu')
    return res

  layers = []

  cur = src if src is not None else _tf.placeholder(dtype=_tf.uint8, shape=(batchSize,) + shape, name='input')
  layers.append(cur)

  cur = _tf.to_float(cur)
  cur = cur / 256.0
  cur = cur - 0.5
  layers.append(cur)

  cur = _Conv('Conv1', cur, count=24, size=5, stride=2)
  layers.append(cur)
  cur = _Conv('Conv2', cur, count=36, size=5, stride=2)
  layers.append(cur)
  cur = _Conv('Conv3', cur, count=48, size=5, stride=2)
  layers.append(cur)
  cur = _Conv('Conv4', cur, count=64, size=5, stride=2)
  layers.append(cur)
  cur = _Conv('Conv5', cur, count=100, size=1, stride=1)
  layers.append(cur)
  cur = _Conv('Conv6', cur, count=1, size=1, stride=1, relu=False)
  layers.append(cur)

  return layers, weights

# This one explicitly pads, then uses "VALID" as the padding mode.
_factor2 = 32
def _buildModel2(rand, src=None, batchSize=1, shape=(64, 64, 3), weights=None):
  assert len(shape) == 3
  assert shape[0] % _factor2 == 0, "Bad size: {}".format(shape)
  assert shape[1] % _factor2 == 0, "Bad size: {}".format(shape)
  assert shape[2] == 3

  if weights is None:
    weights = collections.OrderedDict()
    frozen = False
  else:
    frozen = True

  # Mean and standard deviation for weight initialization.
  mu = 0
  sigma = 0.1

  def _Conv(name, inp, count, size=3, stride=1, relu=True):
    """ Apply a convolution, with the given filter count and size. Uses padding. """
    nameBias = name + '.bias'
    bias = weights.get(nameBias, None)
    if bias is None:
      assert not frozen
      bias = _tf.Variable(_tf.zeros(shape=(count,)), name=nameBias)
      weights[nameBias] = bias
    elif isinstance(bias, np.ndarray):
      bias = _tf.constant(bias, name=nameBias)

    nameKern = name + '.kern'
    kern = weights.get(nameKern, None)
    if kern is None:
      assert not frozen
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

    res = _tf.nn.convolution(input=inp, filter=kern, strides=(stride, stride), padding='VALID', name=name + '.conv')
    res = _tf.nn.bias_add(res, bias, name=name + '.add')
    if relu:
      res = _tf.nn.relu(res, name=name + '.relu')
    return res

  layers = []

  cur = src if src is not None else _tf.placeholder(dtype=_tf.uint8, shape=(batchSize,) + shape, name='input')
  layers.append(cur)

  cur = _tf.to_float(cur)
  cur = cur / 256.0
  cur = cur - 0.5
  layers.append(cur)

  # Now explictly pad by _factor - 1.
  assert _factor2 % 2 == 0
  dz = _factor2 // 2
  cur = _tf.pad(cur, paddings=((0, 0), (dz - 1, dz), (dz - 1, dz), (0, 0)), mode="SYMMETRIC")
  layers.append(cur)

  cur = _Conv('Conv1', cur, count=24, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv2', cur, count=36, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv3', cur, count=48, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv4', cur, count=64, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv5', cur, count=128, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv6', cur, count=100, size=1, stride=1)
  layers.append(cur)
  cur = _Conv('Conv7', cur, count=1, size=1, stride=1, relu=False)
  layers.append(cur)

  return layers, weights


# This one explicitly pads, then uses "VALID" as the padding mode.
_factor3 = 32
def _buildModel3(rand, src=None, batchSize=1, shape=(64, 64, 3), weights=None):
  assert len(shape) == 3
  assert shape[0] % _factor2 == 0, "Bad size: {}".format(shape)
  assert shape[1] % _factor2 == 0, "Bad size: {}".format(shape)
  assert shape[2] == 3

  if weights is None:
    weights = collections.OrderedDict()
    frozen = False
  else:
    frozen = True

  # Mean and standard deviation for weight initialization.
  mu = 0
  sigma = 0.1

  def _Conv(name, inp, count, size=3, stride=1, relu=True):
    """ Apply a convolution, with the given filter count and size. Uses padding. """
    nameBias = name + '.bias'
    bias = weights.get(nameBias, None)
    if bias is None:
      assert not frozen
      bias = _tf.Variable(_tf.zeros(shape=(count,)), name=nameBias)
      weights[nameBias] = bias
    elif isinstance(bias, np.ndarray):
      bias = _tf.constant(bias, name=nameBias)

    nameKern = name + '.kern'
    kern = weights.get(nameKern, None)
    if kern is None:
      assert not frozen
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

    res = _tf.nn.convolution(input=inp, filter=kern, strides=(stride, stride), padding='VALID', name=name + '.conv')
    res = _tf.nn.bias_add(res, bias, name=name + '.add')
    if relu:
      res = _tf.nn.relu(res, name=name + '.relu')
    return res

  layers = []

  cur = src if src is not None else _tf.placeholder(dtype=_tf.uint8, shape=(batchSize,) + shape, name='input')
  layers.append(cur)

  cur = _tf.to_float(cur)
  cur = cur / 256.0
  cur = cur - 0.5
  layers.append(cur)

  # Now explictly pad by _factor - 1.
  assert _factor2 % 2 == 0
  dz = _factor2 // 2
  cur = _tf.pad(cur, paddings=((0, 0), (dz - 1, dz), (dz - 1, dz), (0, 0)), mode="SYMMETRIC")
  layers.append(cur)

  # To allow the net to 'learn' color transformations.
  cur = _Conv('Conv0', cur, count=12, size=1)
  layers.append(cur)

  cur = _Conv('Conv1', cur, count=24, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv2', cur, count=36, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv3', cur, count=48, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv4', cur, count=64, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv5', cur, count=128, size=3, stride=2)
  layers.append(cur)
  cur = _Conv('Conv6', cur, count=100, size=1, stride=1)
  layers.append(cur)
  cur = _Conv('Conv7', cur, count=1, size=1, stride=1, relu=False)
  layers.append(cur)

  return layers, weights

buildModel = _buildModel2
_factor = _factor2

def saveModelWeights(sess, weights):
  wts = collections.OrderedDict()
  for k, v in weights.items():
    w = sess.run(v, feed_dict={})
    assert isinstance(w, np.ndarray)
    print("Trained weight '{}' has shape: {}".format(k, w.shape))
    wts[k] = w
  np.savez('model.npz', **wts)

def loadModel(src=None, shape=(704, 1280, 3)):
  weights = dict(np.load('model.npz'))
  layers, _ = buildModel(None, src=src, batchSize=1, shape=shape, weights=weights)
  return layers

def getModelRectsMultiFunc(scales=(1, 1.5, 2, 3, 4), flip=True):
  assert isinstance(scales, tuple) and len(scales) > 0

  yLimBase = 720 - 128

  sess = _tf.Session()
  fns = []
  for scale in scales:
    if scale == 4:
      # 128 pixels.
      yLimBase = 720 - 128
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(2 * 128, 1280), yLim=yLimBase))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(2 * 128, 1280 - 128), yLim=yLimBase))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(1 * 128, 1280), yLim=yLimBase - 64))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(1 * 128, 1280 - 128), yLim=yLimBase - 64))
    elif scale == 3:
      # 96 pixels.
      yLimBase = 720 - 96
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(3 * 96, 1280 - 32), yLim=yLimBase))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(3 * 96, 1280 - 32 - 96), yLim=yLimBase))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(2 * 96, 1280 - 32), yLim=yLimBase - 48))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(2 * 96, 1280 - 32 - 96), yLim=yLimBase - 48))
    elif scale == 2:
      # 64 pixels.
      yLimBase = 720 - 128
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(4 * 64, 1280), yLim=yLimBase))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(4 * 64, 1280 - 64), yLim=yLimBase))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(3 * 64, 1280), yLim=yLimBase - 32))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(3 * 64, 1280 - 64), yLim=yLimBase - 32))
    elif scale == 1.5:
      # 48 pixels.
      yLimBase = 720 - 128 - 64
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(4 * 48, 1280 - 80), yLim=yLimBase, xOffset=-12))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(4 * 48, 1280 - 80), yLim=yLimBase, xOffset=12))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(3 * 48, 1280 - 80), yLim=yLimBase - 24, xOffset=-12))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(3 * 48, 1280 - 80), yLim=yLimBase - 24, xOffset=12))
    elif scale == 1:
      # 32 pixels.
      yLimBase = 720 - 128 - 64
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(4 * 32, 1280 - 128), yLim=yLimBase))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(4 * 32, 1280 - 128 - 32), yLim=yLimBase))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(4 * 32, 1280 - 128), yLim=yLimBase - 16))
      fns.append(getModelRectsFunc(scale=scale, sess=sess, shapeInpFull=(4 * 32, 1280 - 128 - 32), yLim=yLimBase - 16))
    else:
      # Use defaults.
      fns.append(getModelRectsFunc(scale=scale, sess=sess))

  def _do(src):
    rects = []
    for fn in fns:
      fn(src, rects)
    if flip:
      rects2 = []
      src = src[:, ::-1, :]
      for fn in fns:
        fn(src, rects2)
      for pt0, pt1 in rects2:
        rects.append(((src.shape[1] - pt1[0], pt0[1]), (src.shape[1] - pt0[0], pt1[1])))
    return rects

  return _do

def getModelRectsFunc(scale=1, shapeSrc=(720, 1280), shapeInpFull=(640, 1280), yLim=None, xOffset=0, sess=None):
  # Scale should be half of a positive integer.
  assert scale > 0
  assert int(scale * 2) == scale * 2
  scale2 = int(scale * 2)

  assert 0 < shapeInpFull[0] <= shapeSrc[0]
  assert 0 < shapeInpFull[1] <= shapeSrc[1]
  assert all((2 * x) % (scale2 * _factor) == 0 for x in shapeInpFull), \
    "Mismatch between scale and shapeInpFull: {}".format(tuple(2 * x / scale2 for x in shapeInpFull))

  shapeInp = tuple((2 * x) // scale2 for x in shapeInpFull)

  # Get bounds.
  xMin = (shapeSrc[1] - shapeInpFull[1]) // 2 + xOffset
  assert 0 <= xMin
  xLim = xMin + shapeInpFull[1]
  assert xLim <= shapeSrc[1]

  if yLim is None:
    # Default is to extend to the bottom.
    yLim = shapeSrc[0]
  assert shapeInpFull[0] <= yLim <= shapeSrc[0]
  yMin = yLim - shapeInpFull[0]

  layers = loadModel(shape=shapeInp + (3,))
  for lay in layers:
    print(lay.get_shape())

  g_inp = layers[0]
  g_out = layers[-1]

  shapeOut = tuple(g_out.get_shape().as_list())
  assert shapeOut == (1, shapeInp[0] // _factor, shapeInp[1] // _factor, 1), "Mismatch: {} vs {}".format(shapeOut, shapeInp)
  shapeOut = shapeOut[1:3]
  scaleToSrc = int(_factor * scale)
  print(scaleToSrc)

  # REVIEW shonk: What threshold should we use?
  g_outClip = _tf.cast(_tf.greater(_tf.sigmoid(g_out), 0.98), _tf.float32)
  g_outRes = _tf.reshape(g_outClip, shape=shapeOut)

  if sess is None:
    sess = _tf.Session()

  def _do(src, rects=None):
    assert src.shape == shapeSrc + (3,)

    inp = src[yMin:yLim, xMin:xLim, :]
    assert inp.shape[:2] == shapeInpFull
    if shapeInp != shapeInpFull:
      inp = cv2.resize(inp, shapeInp[::-1])
    assert inp.shape[:2] == shapeInp

    fd = {g_inp: inp[None, :, :, :]}
    res = sess.run(g_outRes, feed_dict=fd)
    assert isinstance(res, np.ndarray)
    assert res.shape == shapeOut

    if rects is None:
      rects = []
    for y in range(res.shape[0]):
      for x in range(res.shape[1]):
        if res[y, x] > 0:
          rects.append((
            (xMin + scaleToSrc * x, yMin + scaleToSrc * y),
            (xMin + scaleToSrc * (x + 1), round(yMin + scaleToSrc * (y + 1)))
          ))

    return rects

  return _do
