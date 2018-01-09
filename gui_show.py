#!/usr/bin/env python

""" Gui app for displaying images from a video or from a directory, and applying various transformations.
"""

# This is based on code that I wrote for other image-related tasks, which is why it's a bit
# more general (in some places) than this project warrants. I used a simpler version on the
# last project (behavioral cloning). This version includes the ability to apply a limited
# set of transforms, as well as to apply the pipeline.

import os
import sys
import logging
import time

import tkinter as _tk
import tkinter.filedialog as _dlg

import cv2
import imageio
import PIL.Image as _img
import PIL.ImageTk as _imgtk
import numpy as np

# The lane finding functionality is the the pipeline.py module.
import pipeline as _pipeline

# pylint: disable=missing-docstring

class Application(_tk.Frame):
  """ The application class. """

  def __init__(self, logger, master=None):
    _tk.Frame.__init__(self, master)

    self._logger = logger

    # Image, canvas and perspective fields.
    self._dy = 0
    self._dx = 0

    self._perspective = None
    self._setImageSize(320, 160)

    # The ImageData object.
    self._data = None

    # Millisecond delay when 'Run' is checked.
    self._delay = 1
    # For writing frames when running.
    self._writer = None

    # Position within the dataset and size of the dataset. Used for scrolling.
    self._pos = -1
    self._count = -1

    self.grid()
    self._createWidgets()

    self._pixels = None
    self._idImage = None
    self._image = None
    self._imagetk = None

    # The lane finding function.
    self._pipeline = None

    # The vehicle detection func and heat map object.
    self._getVehicleRects = None
    self._heatMap = None
    self._idPrev = -1

    # Load the camera un-distort function.
    self._undistort = _pipeline.getUndistortFunc(self._logger)

  def _createWidgets(self):
    """ Create the widgets. """
    padx = 10
    pady = 5

    frame = _tk.Frame(master=self, borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=0)

    _tk.Button(master=frame, text='Quit', command=self.quit).pack(side=_tk.LEFT, padx=padx, pady=pady)
    _tk.Button(master=frame, text='Load Video...', command=self.loadVideo).pack(side=_tk.LEFT, padx=padx, pady=pady)
    _tk.Button(master=frame, text='Load Pictures...', command=self.loadPictures).pack(side=_tk.LEFT, padx=padx, pady=pady)
    _tk.Button(master=frame, text='Save Image...', command=self.saveImage).pack(side=_tk.LEFT, padx=padx, pady=pady)
    _tk.Button(master=frame, text='Save Video...', command=self.saveVideo).pack(side=_tk.LEFT, padx=padx, pady=pady)

    frame = _tk.Frame(master=self, borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=0)

    # To run through the frames.
    self.runVar = _tk.IntVar(master=frame, value=0)
    _tk.Checkbutton(master=frame, variable=self.runVar, text='Run', command=self._runToggle).pack(side=_tk.LEFT, padx=padx, pady=pady)
    _tk.Button(master=frame, text='Reset', command=self._resetState).pack(side=_tk.LEFT, padx=padx, pady=pady)
    self.flipVar = _tk.IntVar(master=frame, value=0)
    _tk.Checkbutton(master=frame, variable=self.flipVar, text='Flip Horizontally', command=self._flipToggle).pack(side=_tk.LEFT, padx=padx, pady=pady)
    # To adjust for camera distortion.
    self.undistortVar = _tk.IntVar(master=frame, value=0)
    _tk.Checkbutton(master=frame, variable=self.undistortVar, text='Undistort', command=self._undistortToggle).pack(side=_tk.LEFT, padx=padx, pady=pady)

    frame = _tk.Frame(master=self, borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=0)

    # Whether to apply the lane finding pipeline.
    self.pipelineVar = _tk.IntVar(master=frame, value=0)
    _tk.Checkbutton(master=frame, variable=self.pipelineVar, text='Lane Finding', command=self._pipelineToggle).pack(side=_tk.LEFT, padx=padx, pady=pady)
    # Whether to use 'sensitive mode'.
    self.sensitiveVar = _tk.IntVar(master=frame, value=0)
    _tk.Checkbutton(master=frame, variable=self.sensitiveVar, text='Sensitive', command=self._sensitiveToggle).pack(side=_tk.LEFT, padx=padx, pady=pady)

    frame = _tk.Frame(master=self, borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=0)

    # Whether to apply the vehicle detection model.
    self.modelVar = _tk.IntVar(master=frame, value=0)
    _tk.Checkbutton(master=frame, variable=self.modelVar, text='Vehicle Detection', command=self._modelToggle).pack(side=_tk.LEFT, padx=padx, pady=pady)
    self.showHeatVar = _tk.IntVar(master=frame, value=0)
    _tk.Checkbutton(master=frame, variable=self.showHeatVar, text='Show Heat', command=self._showHeatToggle).pack(side=_tk.LEFT, padx=padx, pady=pady)
    self.showBoundsVar = _tk.IntVar(master=frame, value=0)
    _tk.Checkbutton(master=frame, variable=self.showBoundsVar, text='Show Bounds', command=self._showBoundsToggle).pack(side=_tk.LEFT, padx=padx, pady=pady)

    # Information about the current image.
    frame = _tk.LabelFrame(master=self, text='Item', borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=pady)

    row = 0

    _tk.Label(master=frame, text='Index:').grid(row=row, column=0, padx=padx, sticky=_tk.E)
    self.indexLabel = _tk.Label(master=frame, text='', anchor=_tk.W)
    self.indexLabel.grid(row=row, column=1, sticky=_tk.W)
    row += 1

    _tk.Label(master=frame, text='Id:').grid(row=row, column=0, padx=padx, sticky=_tk.E)
    self.idLabel = _tk.Label(master=frame, text='', anchor=_tk.W)
    self.idLabel.grid(row=row, column=1, sticky=_tk.W)
    row += 1

    _tk.Label(master=frame, text='Dimensions:').grid(row=row, column=0, padx=padx, sticky=_tk.E)
    self.dimsLabel = _tk.Label(master=frame, text='', anchor=_tk.W)
    self.dimsLabel.grid(row=row, column=1, sticky=_tk.W)
    row += 1

    _tk.Label(master=frame, text='Time:').grid(row=row, column=0, padx=padx, sticky=_tk.E)
    self.timeLabel = _tk.Label(master=frame, text='', anchor=_tk.W)
    self.timeLabel.grid(row=row, column=1, sticky=_tk.W)
    row = None

    frame = _tk.Frame(master=self, borderwidth=3)
    frame.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=pady)

    # For scrolling through the frames/images.
    # Create a sub-frame around the scroll bar and another frame that sets the width.
    # REVIEW shonk: Is there a better way?
    sub = _tk.Frame(master=frame)
    sub.pack(side=_tk.LEFT, padx=padx, pady=pady)
    _tk.Frame(master=sub, height=0, width=400).pack()
    self.dataScroll = _tk.Scrollbar(master=sub, jump=1, orient=_tk.HORIZONTAL, command=self.scrollData)
    self.dataScroll.pack(fill=_tk.X)

    self.dataLabel = _tk.Label(master=frame, text='', anchor=_tk.W, width=20)
    self.dataLabel.pack(side=_tk.LEFT, padx=padx, pady=pady)

    # The image canvas. This can also contain perspective guide lines.
    self.canvas = _tk.Canvas(master=self, height=self._dy, width=self._dx, borderwidth=3)
    self.canvas.pack(fill=_tk.X, side=_tk.TOP, padx=padx, pady=pady)

  def close(self):
    """ Cleanup. """
    self._stop()
    if self._data is not None:
      self._data.close()
      self._data = None

  def quit(self):
    self._logger.debug("Quitting...")
    _tk.Frame.quit(self)

  def loadVideo(self):
    """ Select and load a video. """
    f = _dlg.askopenfilename(
      parent=self, title='Choose a video file', filetypes=(("Video", ".mp4"),))
    if f is None or not os.path.isfile(f):
      return
    data = ImageDataFromVideo(self._logger, f)
    self._setDataSet(data)

  def loadPictures(self):
    """ Select and load a directory of images. """
    d = _dlg.askdirectory(
      parent=self, title='Choose an image directory',
      mustexist=True)
    if d is None or not os.path.isdir(d):
      return
    data = ImageDataFromPictures(self._logger, d)
    self._setDataSet(data)

  def saveImage(self):
    """ Save the current transformed image. """
    self._stop()
    if self._pixels is None:
      return
    f = _dlg.asksaveasfilename(
      parent=self, title='Save Image As', filetypes=(("Image", ".png"),))
    if f is None or f == '':
      return
    root, ext = os.path.splitext(f)
    # Default to .png.
    if ext == '':
      assert root == f
      f += '.png'
    imageio.imwrite(f, self._pixels)

  def saveVideo(self):
    """ Save the current transformed images as a video, starting from the first frame.
    This resets the pipeline, so it gets clean state.
    """
    if self._data is None:
      return
    self._stop()
    f = _dlg.asksaveasfilename(
      parent=self, title='Save Video As', filetypes=(("Video", ".mp4"),))
    if f is None or f == '':
      return

    root, ext = os.path.splitext(f)
    # Default to .mp4
    if ext == '':
      assert root == f
      f += '.mp4'

    # Create the writer. The _setImage method takes care of writing the frames.
    self._writer = imageio.get_writer(f, fps=60)

    # Reset everything, then start playing.
    self._pipeline = None
    self._getVehicleRects = None
    self._heatMap = None

    self.jumpData(0)
    self.runVar.set(1)
    self._runToggle()

  def _setDataSet(self, data):
    assert isinstance(data, ImageData)
    if data.count == 0:
      return
    if self._data is not None:
      self._data.close()
      self._data = None
    # Since the pipeline carries state, reset it.
    self._pipeline = None
    self._getVehicleRects = None
    self._heatMap = None

    self._data = data
    self.jumpData(0)

  @property
  def _running(self):
    return self.runVar.get() != 0

  def _runToggle(self):
    if self._data is not None:
      self._data.setRunning(self._running)
    if self._running:
      self.after(self._delay, self._fetch)
    elif self._writer is not None:
      self._writer.close()
      self._writer = None
      self._logger.info("Closed video writer")

  def _stop(self):
    self.runVar.set(0)
    if self._writer is not None:
      self._writer.close()
      self._writer = None
      self._logger.info("Closed video writer")
    if self._data is not None:
      self._data.setRunning(self._running)

  def _fetch(self):
    """ If we're 'running', advance to the next frame. """
    if not self._running:
      return

    if not self._data.next():
      self._stop()
      return

    self._setImage()
    self.after(self._delay, self._fetch)

  def _setImageSize(self, dy, dx):
    """ Make sure our state matches the given image size. """
    assert isinstance(dy, int) and dy > 0
    assert isinstance(dx, int) and dx > 0

    if dy != self._dy or dx != self._dx:
      self._dy = dy
      self._dx = dx

      # Forget any old pipeline function and get the new perspective.
      self._pipeline = None
      self._getVehicleRects = None
      self._heatMap = None
      self._perspective = _pipeline.Perspective(self._dy, self._dx)

  def _setImage(self):
    # Get the source pixels and shape.
    pixels = self._data.pixels
    dy, dx, _ = pixels.shape

    # Set everything to match this image size.
    self._setImageSize(dy, dx)

    usePipeline = self.pipelineVar.get() != 0
    useModel = self.modelVar.get() != 0

    if usePipeline and self._pipeline is None:
      self._pipeline = _pipeline.getPipelineFunc(pixels.shape, self._perspective, self.sensitiveVar.get() != 0)
    if useModel and self._getVehicleRects is None:
      import model as _model
      scales = (0.5, 1, 1.5, 2)
      self._getVehicleRects = _model.getModelRectsMultiFunc(scales=scales, flip=True)
      self._heatMap = _model.HeatMap()
      self._idPrev = -1

    # Do the transformations, and time it.
    t0 = time.time()

    if self.flipVar.get() != 0:
      pixels = pixels[:, ::-1, :]

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

    if usePipeline:
      pixels = self._drawLaneInfo(pixels, lineInfo)
    if useModel:
      pixels = self._drawVehicleInfo(pixels, rects)

    # Get the total time for the transformations.
    dt = time.time() - t0

    # Add the image to the canvas.
    if self._idImage is not None:
      self.canvas.delete(self._idImage)
      self._idImage = None
    self.canvas.config(height=self._dy, width=self._dx)

    self._pixels = pixels
    self._image = _img.fromarray(pixels, 'RGB')
    self._imagetk = _imgtk.PhotoImage(self._image)

    self._idImage = self.canvas.create_image(self._dx // 2, self._dy // 2, image=self._imagetk)
    # Make sure the guide lines (if present) are on top of the image.
    self.canvas.tag_lower(self._idImage)

    # Position the scroll bar.
    pos = self._data.index
    count = self._data.count
    if count <= 0:
      count = -1
      self.dataLabel.config(text="")
      self.dataScroll.set(0, 1)
    else:
      self.dataLabel.config(text="{0} of {1}".format(pos + 1, count))
      x = pos * 7 / (8.0 * max(1, count - 1))
      self.dataScroll.set(x, x + 0.125)

    self._pos = pos
    self._count = count

    # Set the information labels.
    self.indexLabel.config(text="{}".format(pos))
    self.idLabel.config(text="{}".format(self._data.id))
    self.dimsLabel.config(text="{0} by {1}".format(dx, dy))
    self.timeLabel.config(text="{:6.02f}".format(dt * 1000))

    # Write out the frame if we are saving video.
    if self._writer is not None:
      self._writer.append_data(self._pixels)

  def _drawLaneInfo(self, pixels, lineInfo, printStats=True, drawRaw=False):
    dy, dx, _ = pixels.shape

    # Draw the lane lines.
    if lineInfo.coefsLeft is not None or lineInfo.coefsRight is not None:
      buf = np.zeros_like(pixels)
      ptsList = []

      # Draw them.
      ys = np.linspace(0, pixels.shape[0] - 1, pixels.shape[0])
      ys2 = np.square(ys)
      coefs = lineInfo.coefsLeft
      if coefs is not None:
        xs = coefs[0] * ys2 + coefs[1] * ys + coefs[2]
        pts = np.int32(np.stack((xs, ys[::-1]), axis=1))
        ptsList.append(pts)
      coefs = lineInfo.coefsRight
      if coefs is not None:
        xs = coefs[0] * ys2 + coefs[1] * ys + coefs[2]
        pts = np.int32(np.stack((xs, ys[::-1]), axis=1))
        ptsList.append(pts)

      assert 1 <= len(ptsList) <= 2
      useFill = True
      if useFill and len(ptsList) == 2:
        pts = np.concatenate((ptsList[0], ptsList[1][::-1]), axis=0)
        cv2.fillPoly(buf, [pts], (0, 0xFF, 0))
      else:
        cv2.polylines(buf, ptsList, isClosed=False, color=(0, 0xFF, 0), thickness=15)

      # Draw the raw 'fit' points, using colored circles.
      if drawRaw:
        for i, pts in enumerate((lineInfo.ptsLeft, lineInfo.ptsRight)):
          if pts is None:
            continue
          assert isinstance(pts, tuple) and len(pts) == 2
          assert len(pts[0]) == len(pts[1])
          clr = (0xFF, 0, 0) if i == 0 else (0, 0, 0xFF)
          for x, y in zip(pts[0], pts[1]):
            cv2.circle(buf, (int(x), int(dy - 1 - y)), 5, clr, -1)

      buf = self._perspective.undo(buf)
      pixels = cv2.addWeighted(pixels, 1, buf, 0.3, 0, dst=pixels)

      # Draw the statistics.
      if printStats:
        fnt = cv2.FONT_HERSHEY_DUPLEX
        curv0 = lineInfo.curvatureLeft
        curv1 = lineInfo.curvatureRight
        curvm = lineInfo.curvatureMean
        offset = lineInfo.offset

        ht = 30
        def _fmt(val):
          return "" if val is None else "{:+8.6f}".format(val)
        def _fmtr(val):
          return "" if val is None or abs(val) <= 0.00001 else "{:+8.2f}".format(1.0 / val)

        def _put(row, x, text):
          cv2.putText(pixels, text, (x, row * ht), fnt, 1, (0xFF, 0, 0))
        lft = 10
        mid = dx // 2
        _put(1, lft, " Left curvature: {} / m".format(_fmt(curv0)))
        _put(1, mid, " Left radius: {} m".format(_fmtr(curv0)))
        _put(2, lft, "Right curvature: {} / m".format(_fmt(curv1)))
        _put(2, mid, "Right radius: {} m".format(_fmtr(curv1)))
        _put(3, lft, "Mean curvature: {} / m".format(_fmt(curvm)))
        _put(3, mid, "Mean radius: {} m".format(_fmtr(curvm)))
        _put(4, lft, "Offset: {} m".format(_fmt(offset)))

    return pixels

  def _drawVehicleInfo(self, pixels, rects):
    # This assumes the heat map has already been updated with the rects.
    # The rects are passed here in case we want to draw them.
    if self.showBoundsVar.get() != 0:
      bounds = self._heatMap.getBounds()
      for pt0, pt1 in bounds:
        # Blue for bounds.
        cv2.rectangle(pixels, pt0, pt1, (0, 0, 0xFF), 3)
    elif self.showHeatVar.get() != 0:
      pixels = self._heatMap.render(pixels)
    else:
      for rc in rects:
        # Red for raw rectangles.
        cv2.rectangle(pixels, (rc[0, 0], rc[0, 1]), (rc[1, 0], rc[1, 1]), (0xFF, 0, 0), 1)

    return pixels

  def _undistortToggle(self):
    self._setImage()

  def _resetState(self):
    self._pipeline = None
    self._getVehicleRects = None
    self._heatMap = None
    self._setImage()

  def _flipToggle(self):
    self._resetState()

  def _pipelineToggle(self):
    self._setImage()

  def _overlayToggle(self):
    self._setImage()

  def _sensitiveToggle(self):
    # Toss the current pipeline, so it gets recreated.
    self._pipeline = None
    self._setImage()

  def _modelToggle(self):
    self._setImage()

  def _showHeatToggle(self):
    self._setImage()

  def _showBoundsToggle(self):
    self._setImage()

  def nextData(self):
    self._data.next()
    self._setImage()

  def prevData(self):
    self._data.prev()
    self._setImage()

  def jumpData(self, index):
    self._data.jumpTo(index)
    self._setImage()

  def scrollData(self, kind, value, unit=None):
    count = self._count
    if count <= 0:
      return

    pos = self._pos

    if kind == _tk.SCROLL:
      v = int(value)
      assert v == -1 or v == 1, "Unexpected value: {0}".format(v)
      if unit == _tk.UNITS:
        if v < 0 and pos > 0:
          self.prevData()
        elif v > 0 and pos < count - 1:
          self.nextData()
      elif unit == _tk.PAGES:
        inc = max(1, int(round(count / 10)))
        posNew = max(0, min(count - 1, pos + v * inc))
        if posNew != pos:
          self.jumpData(posNew)
    elif kind == _tk.MOVETO:
      v = float(value)
      posNew = max(0, min(count - 1, int(v * (count - 1) * 8 / 7.0)))
      self.jumpData(posNew)

class ImageData(object):
  """ Base class for an image collection. """
  def __init__(self):
    super(ImageData, self).__init__()

    self._index = -1
    self._id = None
    self._pixels = None
    self._extra = None
    self._seekable = True

  def close(self):
    pass

  def setRunning(self, running):
    # Camera based classes need to know when we are 'running', so need to override this.
    pass

  @property
  def count(self):
    if not self._seekable:
      return -1
    return self._countCore

  @property
  def _countCore(self):
    pass

  @property
  def index(self):
    return self._index

  # pylint: disable=C0103
  @property
  def id(self):
    return self._id

  @property
  def pixels(self):
    return self._pixels

  @property
  def extra(self):
    return self._extra

  def jumpTo(self, index):
    if self._seekable:
      index = max(0, min(self.count - 1, index))
    return self._loadPixels(index)

  def next(self):
    index = self._index + 1
    if self._seekable:
      index = min(self.count - 1, index)
    return self._loadPixels(index)

  def prev(self):
    index = max(0, self._index - 1)
    return self._loadPixels(index)

  def _loadPixels(self, index):
    pass

class ImageDataFromVideo(ImageData):
  """ Loads and wraps a video. """
  def __init__(self, logger, path):
    super(ImageDataFromVideo, self).__init__()

    logger.debug("Loading video: '%s'", path)

    self._logger = logger

    self._reader = imageio.get_reader(path)
    self._count = self._reader.get_length()
    assert self._count > 0, "The video is empty!"

  @property
  def _countCore(self):
    return self._count

  def _loadPixels(self, index):
    assert 0 <= index < self._count
    if index == self._index:
      assert self._pixels is not None
      return False

    raw = self._reader.get_data(index)

    self._pixels = raw.base
    self._id = index
    self._index = index

    return True

class ImageDataFromPictures(ImageData):
  """ Loads images in a directory. The images are loaded lazily (on-demand). """
  def __init__(self, logger, path):
    super(ImageDataFromPictures, self).__init__()

    logger.debug("Loading pictures from: '%s'", path)

    self._logger = logger
    self._dir = None
    self._files = None
    self._findFiles(path)

  def _findFiles(self, path):
    self._dir = path
    self._files = list()
    for name in os.listdir(self._dir):
      if name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png'):
        self._files.append(name)

  @property
  def _countCore(self):
    return len(self._files)

  def _loadPixels(self, index):
    assert 0 <= index < self.count
    if index == self._index:
      assert self._pixels is not None
      return False

    self._logger.debug("Loading: '%s'", self._files[index])
    raw = _img.open(os.path.join(self._dir, self._files[index]))
    if raw.mode != 'RGB':
      self._logger.debug("Converting from %s to %s", raw.mode, 'RGB')
      raw = raw.convert('RGB')

    self._pixels = np.asarray(raw)
    self._id = self._files[index]
    self._index = index

    return True

def _run(logger):
  app = Application(logger)
  app.master.title('Xevo AI Data Viewer')

  # Load initial data set, once the mainloop is spun up.
  app.after(1, app.loadVideo)

  logger.debug('Entering mainloop')
  app.mainloop()
  logger.debug('Exited mainloop')

  app.close()
  logger.debug('Closed')

def initConsoleLogger(loggerName, verbosity=1):
  """ Set up console-based logging and return the logger. """

  logging.basicConfig(
    level=logging.DEBUG if verbosity >= 2 else logging.INFO if verbosity >= 1 else logging.WARN,
    format='[%(asctime)s %(levelname)s %(name)s]  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
  logger = logging.getLogger(loggerName)
  logger.info("Python version: %s", sys.version)

  return logger

def run(_):
  """ Run the script. """
  logger = initConsoleLogger('gui_show', verbosity=1)
  _run(logger)

if __name__ == "__main__":
  run(sys.argv[1:])
