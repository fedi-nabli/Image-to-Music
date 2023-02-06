"""
Created on Mon Feb 06 2023

@author: Fedi Nabli
"""

#Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import random
from pedalboard import Chorus, Reverb, Gain, LadderFilter, Phaser, Delay, PitchShift, Distortion
from pedalboard._pedalboard import Pedalboard
from pedalboard.io import AudioFile
from PIL import Image
from scipy.io import wavfile
import librosa
import glob

# This functions generates frequencies in Hertz from notes
def get_piano_notes():
  # White keys are in uppercasae and sharp keys (blacks) are in lowercase
  octave = ['C', 'c', 'D', 'd', 'E', 'F', 'G', 'g', 'A', 'a', 'B']
  base_freq = 440 # Frequency of not A4
  keys = np.array([x+str(y) for y in range(0, 9) for x in octave])
  
  #Trim to standard 88 keys
  start = np.where(keys == 'A0')[0][0]
  end = np.where(keys == 'C8')[0][0]
  keys = keys[start:end+1]

  note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
  note_freqs[''] = 0.0 #stop
  return note_freqs

# Make scale as specified by the user
def maske_scale(which_octave, which_key, which_scale):
  # Load not distionary
  note_freqs = get_piano_notes()

  # Define tones. Uppercase is white keys and lowercase is black keys
  scale_intervals = ['A', 'a', 'B', 'C', 'c', 'D', 'd', 'E', 'F', 'G', 'g']

  # Find index of desired keys
  index = scale_intervals.index(which_key)

  # Redefine scale intervals so that scale intervals begin with which_key
  new_scale = scale_intervals[index:12] + scale_intervals[:index]
  
  # Choose scale
  scale = []
  if which_scale == 'AEOLIAN':
    scale = [0, 2, 3, 5, 7, 8, 10]
  elif which_scale == 'BLUES':
    scale = [0, 2, 3, 4, 5, 7, 9, 10, 11]
  elif which_scale == 'PHYRIGIAN':
    scale = [0, 1, 3, 5, 7, 8, 10]
  elif which_scale == 'CHROMATIC':
    scale = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  elif which_scale == 'DORIAN':
    scale = [0, 2, 3, 5, 7, 9, 10]
  elif which_scale == 'HARMONIC_MINOR':
    scale = [0, 2, 3, 5, 7, 8, 11]
  elif which_scale == 'LYDIAN':
    scale = [0, 2, 4, 6, 7, 9, 11]
  elif which_scale == 'MAJOR':
    scale = [0, 2, 4, 5, 7, 9, 11]
  elif which_scale == 'MELODIC_MINOR':
    scale = [0, 2, 3, 5, 7, 8, 9, 10, 11]
  elif which_scale == 'MINOR':
    scale = [0, 2, 3, 5, 7, 8, 10]
  elif which_scale == 'MIXOLYDIAN':
    scale = [0, 2, 4, 5, 7, 9, 10]
  elif which_scale == 'NATURAL_MINOR':
    scale = [0, 2, 3, 5, 7, 8, 10]
  elif which_scale == 'PENTATONIC':
    scale = [0, 2, 4, 7, 9]
  else:
    print('Invalid scale name')
  
  # Initialize arrays
  freqs = []
  for i in range(len(scale)):
    note = new_scale[scale[i]] + str(which_octave)
    freq_to_add = note_freqs[note]
    freqs.append(freq_to_add)
  
  return freqs

# Convert Hue value to a frequency
def hue2freq(h, scale_freqs):
  # Initializing variables
  thresholds = [26, 52, 78, 104, 128, 154, 180]
  note = 0

  if (h <= thresholds[0]):
    note = scale_freqs[0]
  elif (h > thresholds[0]) & (h <= thresholds[1]):
    note = scale_freqs[1]
  elif (h > thresholds[1]) & (h <= thresholds[2]):
    note = scale_freqs[2]
  elif (h > thresholds[2]) & (h <= thresholds[3]):
    note = scale_freqs[3]
  elif (h > thresholds[3]) & (h <= thresholds[4]):
    note = scale_freqs[4]
  elif (h > thresholds[4]) & (h <= thresholds[5]):
    note = scale_freqs[5]
  elif (h > thresholds[5]) & (h <= thresholds[6]):
    note = scale_freqs[6]
  else:
    note = scale_freqs[0]

  return note

