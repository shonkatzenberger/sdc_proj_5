""" Script to read the raw .png images from ../Data/vehicles and ../Data/non-vehicles
and save them in ../Data/vehicles.npy and ../Data/non-vehicles.npy, with each file
containing a single numpy array of shape (N, 64, 64, 3) with dtype uint8.
"""

#pylint: disable=C0111
#pylint: disable=C0103

import os
import glob
import imageio
import numpy as np

def loadRawImages(path, shape=None):
  files = glob.iglob(os.path.join(path, '**', '*.png'), recursive=True)
  images = []
  for file in files:
    pixels = imageio.imread(file)
    if shape is None:
      shape = pixels.shape
    elif shape != pixels.shape:
      print("Shape mismatch: {} vs {}".format(shape, pixels.shape))
      continue
    if pixels.dtype != np.uint8:
      print("Converting image '{}' with max {}".format(file, pixels.max()))
      pixels = pixels.astype(np.uint8)
    images.append(pixels)

  print("Read {} images of shape {}".format(len(images), shape))
  return np.stack(images, axis=0)

def _run():
  dataVeh = loadRawImages('../Data/vehicles/vehicles', shape=(64, 64, 3))
  dataNon = loadRawImages('../Data/non-vehicles/non-vehicles', shape=(64, 64, 3))

  np.save('../Data/vehicles.npy', dataVeh)
  np.save('../Data/non-vehicles.npy', dataNon)

if __name__ == "__main__":
  _run()
