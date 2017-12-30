#pylint: disable=C0111
#pylint: disable=C0103

import time
import numpy as np
import tensorflow as tf

import model as _model

def _loadAndSplitData(rand, frac=0.9, useFlips=True):
  dataVeh = np.load('../Data/vehicles.npy')
  dataNon = np.load('../Data/non-vehicles.npy')
  assert len(dataVeh.shape) == 4
  assert len(dataNon.shape) == 4
  assert dataVeh.shape[1:] == dataNon.shape[1:]

  xs = np.concatenate((dataVeh, dataNon), axis=0)
  ys = np.concatenate((np.ones(shape=(dataVeh.shape[0],), dtype=np.float32), np.zeros(shape=(dataNon.shape[0],), dtype=np.float32)))

  assert xs.shape[0] == ys.shape[0]
  num = xs.shape[0]

  # Generate a random permutation of the data.
  indices = rand.permutation(num)
  sub = int(num * frac)
  inds0 = indices[:sub]
  inds1 = indices[sub:]
  xs0, ys0, xs1, ys1 = xs[inds0], ys[inds0], xs[inds1], ys[inds1]

  if useFlips:
    xs0 = np.concatenate((xs0, xs0[:, ::-1, :]), axis=0)
    ys0 = np.concatenate((ys0, ys0), axis=0)
    xs1 = np.concatenate((xs1, xs1[:, ::-1, :]), axis=0)
    ys1 = np.concatenate((ys1, ys1), axis=0)

  return xs0, ys0, xs1, ys1

def _run():
  rand = np.random.RandomState(42)

  layers, weights = _model.buildModel(rand, batchSize=None)

  for k, v in weights.items():
    print("Weight '{}' has shape {}".format(k, v.get_shape()))
  for x in layers:
    print("Layer '{}' has shape {}".format(x.name, x.get_shape()))

  g_x = layers[0]
  g_logits = layers[-1]
  shapeDst = tuple(g_logits.get_shape().as_list())

  g_y = tf.placeholder(tf.float32, (None,))
  g_yRep = tf.tile(tf.reshape(g_y, shape=(-1, 1, 1, 1)), multiples=(1,) + shapeDst[1:])

  # Learning rate placeholder.
  g_lr = tf.placeholder(tf.float32, ())

  # For reporting statistics.
  g_correct = tf.equal(tf.cast(tf.greater(g_logits, 0.0), tf.float32), g_yRep)
  g_correct_sum = tf.reduce_sum(tf.cast(g_correct, tf.int32)) / (shapeDst[1] * shapeDst[2])

  # Apply sigmoid and cross entropy.
  g_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=g_yRep, logits=g_logits)

  # We'll use the loss-sum when reporting statistics, and the loss-mean when training.
  g_loss_sum = tf.reduce_sum(g_ce)
  g_loss_mean = tf.reduce_mean(g_ce)

  # Use Adam with the place-holder learning rate.
  g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lr)
  g_train_step = g_optimizer.minimize(g_loss_mean)

  def _eval(sess, xs, ys, batchSize=64):
    """ Returns the number of correct predictions, sum of losses, and the number of samples.
    The caller can divide the first two by the third to get averages.
    """
    assert batchSize > 0
    assert xs.shape[0] == ys.shape[0]
    num = xs.shape[0]

    acc_sum = 0
    loss_sum = 0.0
    for ii in range(0, num, batchSize):
      xsCur = xs[ii : ii + batchSize]
      ysCur = ys[ii : ii + batchSize]
      acc, loss = sess.run((g_correct_sum, g_loss_sum), feed_dict={g_x: xsCur, g_y: ysCur})
      acc_sum += acc
      loss_sum += loss
    return acc_sum, loss_sum, num

  def _trainOne(sess, xs, ys, rate, batchSize=64):
    """ Performs one training epoch. """
    assert batchSize > 0
    assert xs.shape[0] == ys.shape[0]
    num = xs.shape[0]

    # Generate a random permutation of the training data.
    indices = rand.permutation(num)

    # Run the optimizer on each batch.
    for ii in range(0, num, batchSize):
      inds = indices[ii : ii + batchSize]
      xsCur = xs[inds]
      ysCur = ys[inds]
      sess.run(g_train_step, feed_dict={g_x: xsCur, g_y: ysCur, g_lr: rate})

  def _trainMulti(sess, epochs, xsTrain, ysTrain, xsValid, ysValid, rate, batchSize=64):
    """ Performs multiple epochs, and reports statistics after each. """
    assert epochs > 0

    print("*** Starting {} epochs with batch size {} and learning rate {}".format(epochs, batchSize, rate))
    for i in range(epochs):
      t0 = time.time()
      _trainOne(sess, xsTrain, ysTrain, rate, batchSize)
      t1 = time.time()
      print("  Epoch {} took: {:.04f} sec".format(i, t1 - t0))

      acc, loss, num = _eval(sess, xsTrain, ysTrain, batchSize)
      print("    Training   accuracy and loss: {:.03f}, {:.03f}".format(acc / num, loss / num))
      acc, loss, num = _eval(sess, xsValid, ysValid, batchSize)
      print("    Validation accuracy and loss: {:.03f}, {:.03f}".format(acc / num, loss / num))

  featuresTrain, labelsTrain, featuresValid, labelsValid = _loadAndSplitData(rand, frac=0.9, useFlips=True)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Report the initial statistics (before any epochs). Generally, the accuracy is expected to be around 1 / 2.
    acc, loss, num = _eval(sess, featuresTrain, labelsTrain)
    print("Training   accuracy and loss before: {:.03f}, {:.03f}".format(acc / num, loss / num))
    acc, loss, num = _eval(sess, featuresValid, labelsValid)
    print("Validation accuracy and loss before: {:.03f}, {:.03f}".format(acc / num, loss / num))

    batchSize = 64
    _trainMulti(sess, 5, featuresTrain, labelsTrain, featuresValid, labelsValid, rate=0.00100, batchSize=batchSize)
    _trainMulti(sess, 5, featuresTrain, labelsTrain, featuresValid, labelsValid, rate=0.00030, batchSize=batchSize)
    # _trainMulti(sess, 5, featuresTrain, labelsTrain, featuresValid, labelsValid, rate=0.00010, batchSize=batchSize)
    # _trainMulti(sess, 5, featuresTrain, labelsTrain, featuresValid, labelsValid, rate=0.00005, batchSize=batchSize)

    _model.saveModelWeights(sess, weights)

if __name__ == "__main__":
  _run()
