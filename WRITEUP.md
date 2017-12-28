# Vehicle Detection and Tracking

Shon Katzenberger  
shon@katzenberger-family.com  
December 31, 2017  
October, 2017 class of SDCND term1

## Assignment

Item 41 of the class states:

    In this project, your goal is to write a software pipeline to identify vehicles in a video from a
    front-facing camera on a car. The test images and project video are available in the project
    repository. There is an writeup template in the repository provided as a starting point for your
    writeup of the project.

The writeup template states:

    The goals / steps of this project are the following:

    * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set
      of images and train a Linear SVM classifier.
    * Optionally, you can also apply a color transform and append binned color features, as well
      as histograms of color, to your HOG feature vector.
    * Note: for those first two steps don't forget to normalize your features and randomize a
      selection for training and testing.
    * Implement a sliding-window technique and use your trained classifier to search for vehicles
      in images.
    * Run your pipeline on a video stream (start with the test_video.mp4 and later implement on
      full project_video.mp4) and create a heat map of recurring detections frame by frame to reject
      outliers and follow detected vehicles.
    * Estimate a bounding box for vehicles detected.

The template writeup is assuming use of HOG and SVM. I chose to use neither of those, for reasons explained below.

[//]: # (Image References)

[image00]: ./images/boxes_15.png
[image01]: ./images/boxes_15_quad.png
[image02]: ./images/boxes_all.png
[image03]: ./images/boxes_all_quad.png
[image04]: ./images/raw_rects_00.png
[image05]: ./images/raw_rects_01.png
[image06]: ./images/heat_00.png
[image07]: ./images/heat_01.png
[image08]: ./images/bounds_00.png
[image09]: ./images/bounds_01.png
[image10]: ./images/test_image_boxes.png
[image11]: ./images/test_image_heat.png
[image12]: ./images/gui_show.png

[image20]: ./images/extra1630.png
[image21]: ./images/extra1633.png
[image22]: ./images/extra2065.png
[image23]: ./images/extra3009.png
[image24]: ./images/extra5061.png
[image25]: ./images/extra5121.png

## Submission Details

The submission consists of the following files:
* `WRITEUP.md`: This file.
* `consolidate_data.py`: The script to read the images in the `Data/vehicles` and `Data/non-vehicles` folders and save them
as two `numpy` arrays in `Data/vehicles.npy` and `Data/non-vehicles.npy`. The `numpy` arrays have shape of the form
`(N, 64, 64, 3)`, where `N` is the number of images. This script can be executed directly.
* `train.py`: The script to train the vehicle detection (classification) model. This script can be executed directly.
* `model.npz`: The trained model weights (output of `train.py`).
* `model.py`: The vehicle detection model code. This ***cannot*** be executed directly, but is imported by `train.py` and by
the harness, `gui_show.py`.
* `gui_show.py`: The visualization and harness application. This is a modified version of the harness application I wrote for
the advanced lane finding project.
* `pipeline.py`, `calibrate.py`, `cameraCalibration.p`: These are copied from the advanced lane finding project. There is
one small change in `pipeline.py`, namely that applying `undistort` has been moved out of `pipeline.py`, since the undistorted
image needs to be passed to both the lane finding and the vehicle detection code.
* `videos/vehicle.mp4`: Video of vehicle detection executed on `project_video.mp4`.
* `videos/combined.mp4`: Video of both lane finding and vehicle detection executed on `project_video.mp4`.
* `images`: The directory containing images referenced by this writeup.

Note that all code was written and executed under Windows 10, so may perform slightly differently on a different OS.

## Detection Model Details

### Classifier Details

After experimenting with HOG features a bit, I realized that using HOG would not provide reasonable performance. My target was
to process at least 10 frames per second on my MSI laptop (with GTX 1070 GPU), but applying HOG to a single `720 x 1280`
image took over two seconds! Of course, I would be applying HOG to about a third of the image, but would also need to apply it
for multiple scales of the image, so clearly HOG would not get near 10 frames per second. I wanted to harness the power of the
GPU, so decided to craft a solution using Tensorflow.

Given enough training data, a neural network can easily "learn" gradient features directly from raw input data, so hand crafting
gradient features (such as HOG) is unnecessary.

Ideally, the windowing logic should also be kept on the GPU. The key insight is that a fully convolutional neural network
(with no fully connected layers) can be applied to images of various sizes and automatically performs windowing. That is,
the network can be trained on `64 x 64` images and then applied to much larger images to produce a grid of predictions,
rather than a single prediction.

The neural network I settled on consists of five `3 x 3` convolutions followed by two `1 x 1` convolutions. The `1 x 1`
convolutions play the role of the typical fully connected layers, without locking in the input and output sizes. The `3 x 3`
convolutions use `stride=2` (when training). Since there are five such layers, this network has a spatial reduction factor
of `2^5 = 32`. So applying the network to a `64 x 64` training image produces a `2 x 2` output. Note that the *support*
(region of influence) for an output is larger than `32 x 32` because of the `3 x 3` kernels. For this network, the
support of one output is actually `63 x 63`. When applied to a training image, much of the `63 x 63` support is padding.
I chose to explicitly pad the initial input (using symmetric mode), rather than pad at each layer. The ideal would be to
use training images that are padded with context, but such images weren't available, so padding symmetrically is a reasonable
approximation.

More precisely, the network consists of:
* The input placeholder of shape `(H, W, 3)` and type `uint8`. The shapes of the other layers naturally depend on `H` and `W`.
* Conversion to `float32` and the affine mapping from the half-open interval `[0, 256)` to `[-0.5, 0.5)`, that is,
divide by `256` and subtract `0.5`.
* Symmetric padding of `31` additional cells in each spatial dimension.
* `3 x 3` convolution with `relu` activation, stride `2` and `24` channels.
* `3 x 3` convolution with `relu` activation, stride `2` and `36` channels.
* `3 x 3` convolution with `relu` activation, stride `2` and `48` channels.
* `3 x 3` convolution with `relu` activation, stride `2` and `64` channels.
* `3 x 3` convolution with `relu` activation, stride `2` and `128` channels.
* `1 x 1` convolution with `relu` activation, stride `1` and `100` channels.
* `1 x 1` convolution with no activation, stride `1` and `1` channel. This is the raw "logit" output.

Note that this network is likely higher capacity than is needed, but it accomplished the goal.

For training, I applied sigmoid activation and cross entropy loss with the label being a `2 x 2` tensor of ones (for vehicles)
or zeros (for non-vehicles).

The code for constructing the classifier model is in the `buildModel` function in `model.py`. Here is the core model
definition code:

```
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
```

### Windowing Details

For prediction, I apply the CNN on images of various scales and sizes (detailed below). For this discussion, suppose
the network is to be applied to a sub-image of size `256 x 1280`. Then the output will be `8 x 40`. An output cell with indices
`(i, j)` is interpreted as the prediction for the input rectangle `((32 * i, 32 * j), (32 * i + 32, 32 * j + 32)`.
This provides a prediction every 32 pixels, which is too coarse; we'd like predictions at finer intervals, like every 16 or every
8 pixels. Getting 16 pixels is quite easy: set the stride of the last `3 x 3` convolution to `1` rather than `2`. This increases
the number of outputs to `15 x 79` with output cell `(i, j)` corresponding to the input rectangle
`((16 * i, 16 * j), (16 * i + 32, 16 * j + 32)`.

To get to 8 pixel granularity, we can repeat the trick, but need to apply a twist: set the stride of the last *two*
`3 x 3` convolutions to `1` rather then `2` and set the *dilation* of the last `3 x 3` convolution to `2` rather than `1`.
This produces an output of size `29 x 157` with output cell `(i, j)` corresponding to the input rectangle
`((8 * i, 8 * j), (8 * i + 32, 8 * j + 32)`.

This technique can be extended to provide even finer granularity, but I stopped at quadrupling. Note that the kernel sizes
and values are all maintained by this technique. Only the stride and dilation are changed. See the `buildModel` function
in `model.py` (or the excerpt above) for the model details, and see the `_do` function nested in the `getModelRectsFunc`
function for the prediction details. Both `buildModel` and `getModelRectsFunc` take a `sampleMode` parameter, which
is restricted to the values 1, 2, and 4, with the corresponding granularities being 32, 16, and 8 pixels, respectively.

Note that an increased `sampleMode` increases processing time, but much of the computation for the extra outputs is shared
with the computation needed for the original outputs. That is, doubling the `sampleMode` does not double the amount of
computation. For example, with `scale=1`, the `sess.run` invocation in `_do` costs about 2 ms per frame, regardless
of the value of `sampleMode`. However, the post processing tends to be closer to linear in `sampleMode` and also takes
the bulk of the processing time. With only `scale=1` active, the total processing time per frame (on `test_video.mp4`)
is about 35 ms with `sampleMode=1` and 40 ms with `sampleMode=4`. With all four chosen scales active (see below), the
total processing time per frame (on `test_video.mp4`) is about 55 ms with `sampleMode=1` and 80 ms with `sampleMode=4`.

In hindsight, I should have set `sampleMode=4` when training. Doing so would effectively augment the training data to include
"shifted" versions of the training images. I'll likely experiment with this after submission.

In summary, using a convolutional neural network like this takes care of both feature generation and windowing,
and, most importantly, keeps the computation on the GPU, where it belongs.

### Scales

I experimented with various scaling factors, namely, `0.5, 1.0, 1.5, 2.0, 3.0, 4.0`. Note that scales other than 1.0
require resizing (a portion of) the image. Scale 0.5 grows the image while the scales larger than 1.0 shrink the image.

In the end, I used the first four of these scales, namely `0.5, 1.0, 1.5, 2.0`.

For each scale, I manually chose the image sub-region to use. For example, here are the
boxes (each of size `48 x 48`) for `scale=1.5` with `sampleMode=1`:

![alt text][image00]

Here are the boxes (each of size `48 x 48`) for `scale=1.5` with `sampleMode=4`:

![alt text][image01]

Note that each box is of size `48 x 48` but the boxes overlap every 12 pixels, creating the illusion of smaller boxes.

Here are the boxes for all the scales that I used, with `sampleMode=1`:

![alt text][image02]

Here are the boxes for all the scales that I used, with `sampleMode=4`:

![alt text][image03]

In addition to using multiple scales, I also applied the classifier to the horizontally flipped image. See the
`_do` function nested in the `getModelRectsMultiFunc` function in `model.py`:

```
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
```

### Color Space

I chose to train and predict using the Lab color space. This gave better accuracy and generalization than using RGB.
Note that HSV and HLS are not appropriate since the H component is a piece-wise, cyclic value. That is, comparing two
H values is not really meaningful unless the two values happen to be in the same sub-range of the H space.

### Training Details

The training code is in `train.py`. It is very similar to code that I wrote for the behavioral cloning project.
I used the provided training data, together with a small number of additional non-vehicle examples.
I augmented this data with the horizontally flipped images (see `_loadAndSplitData`):

```
  # Generate a random permutation of the data.
  indices = rand.permutation(num)
  sub = int(num * frac)
  inds0 = indices[:sub]
  inds1 = indices[sub:]
  xs0, ys0, xs1, ys1 = xs[inds0], ys[inds0], xs[inds1], ys[inds1]

  if useFlips:
    xs0 = np.concatenate((xs0, xs0[:, :, ::-1, :]), axis=0)
    ys0 = np.concatenate((ys0, ys0), axis=0)
    # REVIEW shonk: Any reason to also augment the validation set?
    xs1 = np.concatenate((xs1, xs1[:, :, ::-1, :]), axis=0)
    ys1 = np.concatenate((ys1, ys1), axis=0)
```

I split the data with 90% used for training and the remaining 10% used for validation. I trained 5 epochs with
learning rate 0.0010, 3 epochs with learning rate 0.0003, and 3 epochs with learning rate 0.0001. I didn't bother
with dropout or other regularization techniques.

The trained weights are saved in the `model.npz` file. Here's an excerpt:

```
    batchSize = 64
    _trainMulti(sess, 5, featuresTrain, labelsTrain, featuresValid, labelsValid, rate=0.00100, batchSize=batchSize)
    _trainMulti(sess, 3, featuresTrain, labelsTrain, featuresValid, labelsValid, rate=0.00030, batchSize=batchSize)
    _trainMulti(sess, 3, featuresTrain, labelsTrain, featuresValid, labelsValid, rate=0.00010, batchSize=batchSize)
    # _trainMulti(sess, 5, featuresTrain, labelsTrain, featuresValid, labelsValid, rate=0.00005, batchSize=batchSize)

    _model.saveModelWeights(sess, weights)
```

### Prediction Details

The `gui_show.py` harness invokes the `getModelRectsMultiFunc` function in `model.py`, which returns a function
that accepts a `720 x 1280` RGB image and returns a (possibly empty) list of (raw) rectangles. The returned function is
stored in the `self._getVehicleRects` field of the `Application` object. The application also stores, in the
`self._heatMap` field, an instance of the `HeatMap` class in `model.py`.

The code to process an image is in the `_setImage` method in `gui_show.py`:
```
    if usePipeline or useModel or self.undistortVar.get() != 0:
      pixels = self._undistort(pixels)

    if usePipeline:
      _, lineInfo = self._pipeline(pixels)
    if useModel:
      rects = self._getVehicleRects(pixels)
      # Only update the heap map if the frame index has changed.
      # Otherwise, flipping between view modes would change the heat map.
      index = self._data.index
      if self._idPrev != index:
        self._heatMap.update(rects)
        self._idPrev = index
```

To process an image, the application applies distortion correction, then (optionally) invokes the lane finding code,
`self._pipeline`, then (optionally) invokes `self._getVehicleRects`, which returns a numpy array of raw rectangles.
The rectangles are then sent to the heat map object:
```
        self._heatMap.update(rects)
```

### HeatMap Details

The `HeatMap` class is in the `model.py` file. It accumulates information over multiple frames and can provide
either a rendering of the heat map or bounding boxes for detections.

The heat map constructor accepts a tuple of frame weights, defaulting to `(10, 10, 8, 8, 6, 6, 4, 4, 2, 2)`.
The length of this tuple determines the maximum number of active frames in the heat map. The heat map has a queue
containing the rectangles for the active frames. When a new frame is added, if the number of active frames is
already at the maximum, the oldest frame is dropped. All other frames are shifted in the queue and the new frame
is added to the queue. The heat map maintains the current total heat for all the "pixels" in the map, with each
frame rectangle contributing the corresponding frame weight worth of heat for the pixels contained in the rectangle.
Note that since the weights are not necessarily all the same, the amount of heat for each rectangle is updated
whenever a new frame is added (via an invocation of the `update` method). Here's the code for the `update` method:

```
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
    # print("New Max: {}".format(self._heat.max()))
```

Invoking the `getBounds` method of the `HeatMap` object returns a (possibly empty) list of bounding rectangles.
The bounding rectangles are generated by ignoring any heat values below a "low threshold" which is 10 times the
sum of the frame weights, then invoking the `label` function of `scipy.ndimage.measurements`, and then
finding the maximum and minimum extents of each "blob". I drop any rectangle that is less than 32 pixels in either
direction, as well as any blob whose maximum is less than a "high threshold", namely 20 times the sum of the
frame weights. Ideally, the constants 10 and 20 would be settable as they certainly depend on the number of scales
that are used to generate raw rectangles. Here's the `getBounds` code:

```
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
        # print("Rejected for size: {}".format(rc))
        continue

      # Ignore blobs whose max is too small.
      tmp = self._heat[labels == i]
      hi = tmp.max()
      if hi < self._threshHi:
        # print("Rejected for max: {}, {}".format(rc, hi))
        continue

      # print("  Min/max for {} is {}/{}".format(rc, tmp.min(), hi))
      rects.append(rc)

    return rects
```

Note that using multiple scales, smoothing over multiple frames, and using heat thresholds reduces the chance
of false positives.

Here are renderings of the raw rectangles on the first two frames of `test_video.py`:

![alt text][image04]

![alt text][image05]


Here are the corresponding heat map renderings:

![alt text][image06]

![alt text][image07]

Note that the second heat map image is "fuller" since it is an accumulation of the raw rectangles of both frames.

Here are the resulting bounding boxes:

![alt text][image08]

![alt text][image09]

Note that the first frame shows only one bounding box, since the second blob does not meet the "high threshold",
while the accumulated heat map (over the first two frames) *does* meet the threshold.

---

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
### Writeup / README

This file is it!

---
### Histogram of Oriented Gradients (HOG)

As explained above, I did not use HOG directly. A CNN is capable of learning gradient/derivative-like features on its own from
a large enough training dataset. The HOG implementation suggested by the lectures is too slow to be useful for real time
prediction. Using HOG is essentially hand crafting convolutional kernels, which, with recent advances in DNN on GPU technology,
is usually not a productive strategy.

See above for a description of the model and techniques that I used.

---
### Sliding Window Search

As explained above, using a pure convolutional neural network (with no fully connected layers), as well as clever combinations
of stride and dilation effectively implement windowing within the CNN, keeping the related computation on the GPU.

See above for a description of the scales and rectangles that are "searched" using the CNN.

Here's the first example image with resulting raw rectangles, and the associated heat map:

![alt text][image10]

![alt text][image11]

---
### Video Implementation

Here's a [link to the vehicle detection video](./videos/vehicle.mp4), and here's the
[combined lane finding and vehicle detection video](./videos/combined.mp4). These are in the `videos` folder.

The videos are recorded at 60 frames per second. Generation of the videos took roughly 60 - 80 ms and 120 - 140 ms per frame,
respectively (12 - 16 fps and 7 - 8 fps, respectively). To easily view individual frames, I suggest using `python ./gui_show.py`,
and either check the `Run` checkbox, or use the scroll bar:

![alt text][image12]

As explained above, the `HeatMap` class implements combining of overlapping raw rectangles, combining information across
multiple frames, as well as thresholding. All of these help avoid false positives.

---
### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Once I settled on using a convolutional neural network, one issue was how to achieve fine granularity (overlapping prediction rectangles).
Initially, I invoked the CNN on multiple offset sub-regions, but realized that this was repeating a lot of computation. The key insight was
that changing the stride and dilation of the last couple `3 x 3` convolutions was the right thing to do.

Another issue was the quality of the classifier. I had to add some negative examples to fix a couple issues. For example,
the bush on the left of the first frame of `project_video.mp4` and a section of barrier further into the video
were both troublesome false positives. There is also a section of video where the classifier struggles to recognize
the white car (starting around frame 600). I believe this is because the non-vehicle/Extras folder includes a lot of
images of the grassy slope, as well as images containing part of the white car, such as these:

![alt text][image20] ![alt text][image21] ![alt text][image22] ![alt text][image23] ![alt text][image24] ![alt text][image25]

These kinds of issues can easily be improved with additional training data.

Another thing that would likely have helped improve the classifier is to set sampleMode to 2 or 4 while training, as noted in
the discussion above.

Like many of the projects in this class, one can spend endless time tweaking thresholds and parameters, such as the image
scales to use, the sub-region of the image for each scale, the heat map thresholds, the raw classifier threshold, the
minimum bounding rectangle size, the number of frames to use in the heat map, the heat map frame weights,the neural
network architecture, etc. The choices I made were driven by the particular `project_video.mp4` and would likely need
to be further refined / tweaked for other scenarios. For example, with two-way traffic or cross traffic, 10 frames
for the heat map may be too many, since the relative velocity of the other vehicles may be too large to expect 10
frames of continuity.

The most obvious way to improve all of this is to leverage some of the recent advances in object detection,
such as [YOLO](https://pjreddie.com/darknet/yolo/), [SSD](https://arxiv.org/abs/1512.02325), and
[Focal Loss](https://arxiv.org/abs/1708.02002). I personally need to spend more time investigating what's out there.
Perhaps this course could include some resources summarizing techniques like these.
