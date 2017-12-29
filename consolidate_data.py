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
    images.append(pixels)

  print("Read {} images of shape {}".format(len(images), shape))
  return np.stack(images, axis=0)

def loadData():
  dataVeh = np.load('../Data/vehicles.npy')
  dataNon = np.load('../Data/non-vehicles.npy')
  assert len(dataVeh.shape) == 4
  assert len(dataNon.shape) == 4
  assert dataVeh.shape[1:] == dataNon.shape[1:]
  return dataVeh, dataNon

def _run():
  dataVeh = loadRawImages('../Data/vehicles/vehicles', shape=(64, 64, 3))
  dataNon = loadRawImages('../Data/non-vehicles/non-vehicles', shape=(64, 64, 3))

  np.save('../Data/vehicles.npy', dataVeh)
  np.save('../Data/non-vehicles.npy', dataNon)

if __name__ == "__main__":
  _run()
