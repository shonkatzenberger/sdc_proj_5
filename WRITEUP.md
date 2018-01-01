# Vehicle Detection and Tracking

Shon Katzenberger  
shon@katzenberger-family.com  
December 31, 2017  
October, 2017 class of SDCND term1

## Assignment

Item 41 of the class states:

    In this project, your goal is to write a software pipeline to identify vehicles in a video from a front-facing camera
    on a car. The test images and project video are available in the project repository. There is an writeup template in the
    repository provided as a starting point for your writeup of the project.

The writeup template states:

    The goals / steps of this project are the following:

    * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train
      a classifier Linear SVM classifier.
    * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color,
      to your HOG feature vector.
    * Note: for those first two steps don't forget to normalize your features and randomize a selection for training
      and testing.
    * Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
    * Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4)
      and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
    * Estimate a bounding box for vehicles detected.

The template writeup is assuming use of HOG and SVM. I chose to use neither of those, for reasons explained below.

[//]: # (Image References)

[image00]: ./images/boxes_15.png
[image01]: ./images/boxes_15_quad.png
[image02]: ./images/boxes_all.png
[image03]: ./images/boxes_all_quad.png
[image04]: ./images/raw_rects.png
[image05]: ./images/heat.png
[image06]: ./images/bounds.png
[image07]: ./images/test_image_boxes.png
[image08]: ./images/test_image_heat.png
[image09]: ./images/gui_show.png

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Submission Details

The submission consists of the following files:
* `WRITEUP.md`: This file.
* `consolidate_data.py`: The script to read the images in the `Data/vehicles` and `Data/non-vehicles` folders and save them
as two `numpy` arrays in `Data/vehicles.npy` and `Data/non-vehicles.npy`. The `numpy` arrays have shape of the form
`(N, 64, 64, 3)`, where `N` is the number of images. This script can be executed directly.
* `train.py`: The script to model the vehicle detection (classification) model. This script can be executed directly.
* `model.py`: The vehicle detection model code. This ***cannot*** be executed directly, but is imported by `train.py` and by
the harness, `gui_show.py`.
* `model.npz`: The trained model weights.
* `gui_show.py`: The visualization and harness application. This is a modified version of the harness application I used in
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
gradient features in unnecessary. For this project, the provided training data was adequate to produce a good DNN classifier.

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
divide be `256` and subtract `0.5`.
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

The code for constructing the classifier model is in the `buildModel` function in `model.py`.

### Windowing Details

For prediction, I apply the neural network on images of various scales and sizes (detailed below). For this discussion, suppose
the network is to be applied to a sub-image of size `256 x 1280`. Then the output will be `8 x 40`. An output cell with indices
`(i, j)` is interpreted as the prediction for the input rectangle `((32 * i, 32 * j), (32 * (i + 1), 32 * (j + 1))`.
This provides a prediction every 32 pixels, which is too coarse; we'd like predictions at finer intervals, like every 16 or every
8 pixels. Getting 16 pixels is quite easy: set the stride of the last `3 x 3` convolution to `1` rather than `2`. This increases
the number of outputs to `15 x 79` with output cell `(i, j)` corresponding to the input rectangle
`((16 * i, 16 * j), (16 * (i + 2), 16 * (j + 2))`.

To get to 8 pixel granularity, we can repeat the trick, but need to apply a twist: set the stride of the last *two*
`3 x 3` convolutions to `1` rather then `2` and set the *dilation* of the last `3 x 3` convolution to `2` rather than `1`.
This produces an output of size `29 x 157` with output cell `(i, j)` corresponding to the input rectangle
`((8 * i, 8 * j), (8 * (i + 4), 8 * (j + 4))`.

This technique can be extended to provide even finer granularity, but I stopped at quadrupling. Note that the kernel sizes
and values are all maintained by this technique. Only the stride and dilation are changed. See the `buildModel` function
in `model.py` for the model details, and see the `_do` function nested in the `getModelRectsFunc` function for the
prediction details. Both `buildModel` and `getModelRectsFunc` take a `sampleMode` parameter, which is restricted to
the values 1, 2, and 4, with the corresponding granularities being 32, 16, and 8 pixels, respectively.

Note that an increased sampleMode increases processing time, but much of the computation for the extra outputs is shared
with the computation needed for the original outputs. That is, doubling the sampleMode does not double the amount of
computation. For example, with `scale=1`, the `sess.run` invocation in `_do` costs about 2 ms per frame, regardless
of the value of `sampleMode`. However, the post processing tends to be closer to linear in the number of samples and
also takes the bulk of the processing time. With `scale=1` active, the total time to process and render the result is
about 35 ms with `sampleMode=1` and 50 ms with `sampleMode=4`. With all four chosen scales active (see below), the
total processing time per frame is about 65 ms with `sampleMode=1` and 120 ms with `sampleMode=4`.

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

In addition to using multiple scales, I also applied the classifier to the horizontally flipped image.

### Color Space

I chose to train and predict using the Lab color space. This gave better accuracy and generalization than using RGB.
Note that HSV and HLS are not appropriate since the H component is a piece-wise, cyclic, non-linear value.

### Training Details

The training code is in `train.py`. It is very similar to code that I wrote for the behavioral cloning project.
I used the provided training data, together with a small number of additional non-vehicle examples.
I augmented this data with the horizontally flipped images (see `_loadAndSplitData`). I split the data with 90% used for
training and the remaining 10% used for validation. I trained 5 epochs with learning rate 0.0010, 3 epochs with learning rate
0.0003, and 3 epochs with learning rate 0.0001. I didn't bother with dropout or other regularization techniques.

The trained weights are saved in the model.npz file.

### Prediction Details

The `gui_show.py` harness invokes the `getModelRectsMultiFunc` function in `model.py`, which returns a function
that accepts a `720 x 1280` RGB image and returns a (possible empty) list of (raw) rectangles. The returned function is
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
`self._pipeline`, then (optionally) invokes `self._getVehicleRects`, which returns a list of raw rectangles.
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
whenever a new frame is added (via an invocation of the `update` method).

Invoking the `getBounds` method of the `HeatMap` object returns a (possible empty) list of bounding rectangles.
The bounding rectangles are generated by ignoring any heat values below a "low threshold" which is 10 times the
sum of the frame weights, then invoking the `label` function of `scipy.ndimage.measurements`, and then
finding the maximum and minimum extents of each "blob". I drop any rectangle that is less than 48 pixels in either
direction, as well as any blob whose maximum is less than a "high threshold", namely 20 times the sum of the
frame weights. Ideally, the constants 10 and 20 would be settable as they certainly depend on the number of scales
that are used to generate raw rectangles.

Note that using multiple scales, smoothing over multiple frames, and using heat thresholds, reduces the chance
of false positives.

Here's a rendering of the raw rects on the first frame of `test_video.py`:

![alt text][image04]

Here's the corresponding heat map rendering:

![alt text][image05]

Her's the resulting bounding boxes:

![alt text][image06]

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

![alt text][image07]

![alt text][image08]

---
### Video Implementation

Here's a [link to the vehicle detection video](./videos/vehicle.mp4), and here's the
[combined lane finding and vehicle detection video](./videos/combined.mp4). These are in the `videos` folder.

The videos are recorded at 60 frames per second. Generation of the videos took roughly 100 ms and 200 ms per frame,
respectively (10 fps and 5 fps, respectively). To easily view individual frames, I suggest using `python ./gui_show.py`,
and either checking the `Run` checkbox, or use the scroll bar:

![alt text][image09]

As explained above, the `HeatMap` class implements combining of overlapping raw rectangles, combining information across
multiple frames, as well as thresholding. All of these help avoid false positives.

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
