#pylint: disable=C0111
#pylint: disable=C0103

import collections
import cv2
import numpy as np
import scipy.misc as _misc
import tensorflow as _tf

_factor0 = 32
def _buildModel0(rand, src=None, batchSize=1, shape=(64, 64, 3), weights=None):
  assert len(shape) == 3
  assert shape[0] % _factor0 == 0, "Bad size: {}".format(shape)
  assert shape[1] % _factor0 == 0, "Bad size: {}".format(shape)
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
  cur = _Conv('Conv5', cur, count=128, size=5, stride=2)
  layers.append(cur)
  cur = _Conv('Conv6', cur, count=100, size=1, stride=1)
  layers.append(cur)
  cur = _Conv('Conv7', cur, count=1, size=1, stride=1, relu=False)
  layers.append(cur)

  return layers, weights

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

def getModelFunc1(scale=4):
  assert isinstance(scale, int) and 1 <= scale <= 8
  shapeSrc = (720, 1280)
  shapeInpFull = (704, 1280)
  shapeInp = tuple(scale * (x // 4) for x in shapeInpFull)
  yLim = 720
  yMin = yLim - shapeInpFull[0]

  layers = loadModel(shape=shapeInp + (3,))
  for lay in layers:
    print(lay.get_shape())

  g_inp = layers[0]
  g_out = layers[-1]

  shapeTmp = tuple(g_out.get_shape().as_list())
  assert shapeTmp == (1, shapeInp[0] // _factor, shapeInp[1] // _factor, 1), "Mismatch: {} vs {}".format(shapeTmp, shapeInp)

  # REVIEW shonk: What threshold should we use?
  # g_outClip = _tf.maximum(_tf.sigmoid(g_out) - 0.95, 0.0)
  g_outClip = _tf.cast(_tf.greater(_tf.sigmoid(g_out), 0.95), _tf.float32)
  g_outRes = _tf.reshape(g_outClip, shape=shapeTmp[1:3])

  sess = _tf.Session()

  def _do(inp):
    assert inp.shape == shapeSrc + (3,)

    x = inp[yMin:yLim, :, :]
    assert x.shape[:2] == shapeInpFull
    if shapeInp != shapeInpFull:
      # x = _misc.imresize(x, shapeInp)
      x = cv2.resize(x, shapeInp[::-1])

    fd = {g_inp: x[None, :, :, :]}
    res = sess.run(g_outRes, feed_dict=fd)
    assert isinstance(res, np.ndarray)
    assert res.shape == shapeTmp[1:3]

    # res = _misc.imresize(res, shapeInpFull)
    res = cv2.resize(res, shapeInpFull[::-1])
    res = np.reshape(res, res.shape[:2])
    res = np.pad(res, ((yMin, shapeSrc[0] - yLim), (0, 0)), 'constant')

    res[592:, :] = 0
    return res

  return _do

def getModelFunc2(scale=4):
  assert isinstance(scale, int) and 1 <= scale <= 8
  shapeSrc = (720, 1280)
  shapeInpFull = (704, 1280)
  shapeInp = tuple(scale * (x // 4) for x in shapeInpFull)
  yLim = 720
  yMin = yLim - shapeInpFull[0]

  # REVIEW shonk: TF's resize_images doesn't seem to work well when growing the image size.
  g_raw = _tf.placeholder(dtype=_tf.uint8, shape=(1,) + shapeInpFull + (3,))
  g_scaled = g_raw if shapeInpFull == shapeInp else _tf.image.resize_images(g_raw, size=shapeInp + (3,))

  layers = loadModel(src=g_scaled, shape=shapeInp + (3,))
  for lay in layers:
    print(lay.get_shape())

  g_out = layers[-1]

  shapeTmp = tuple(g_out.get_shape().as_list())
  assert shapeTmp == (1, shapeInp[0] // _factor, shapeInp[1] // _factor, 1), "Mismatch: {} vs {}".format(shapeTmp, shapeInp)

  # REVIEW shonk: What threshold should we use?
  # g_outClip = _tf.maximum(_tf.sigmoid(g_out) - 0.95, 0.0)
  g_outClip = _tf.cast(_tf.greater(_tf.sigmoid(g_out), 0.95), _tf.float32)
  g_outRes = _tf.reshape(_tf.image.resize_images(g_outClip, size=shapeInpFull), shape=shapeInpFull)
  g_outPad = _tf.pad(g_outRes, _tf.constant(((yMin, shapeSrc[0] - yLim), (0, 0))))

  sess = _tf.Session()

  def _do(inp):
    assert inp.shape == shapeSrc + (3,)

    x = inp[yMin:yLim, :, :]
    assert x.shape[:2] == shapeInpFull

    fd = {g_raw: x[None, :, :, :]}
    res = sess.run(g_outPad, feed_dict=fd)
    assert isinstance(res, np.ndarray)
    assert res.shape == shapeSrc, "Unexpected shape: {} vs {}".format(res.shape, shapeSrc)
    res[592:, :] = 0
    return res

  return _do
