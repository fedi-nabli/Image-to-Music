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

# Make song from image
def img2music(img, scale = [220.00, 246.94 ,261.63, 293.66, 329.63, 349.23, 415.30],
              sr = 22050, T = 0.1, n_pixels = 60, use_octaves = True, random_pixels = False,
              harmonize = 'U0'):
  """
  Args:
    img    :     (array) image to process
    scale  :     (array) array containing frequencies to map H values to
    sr     :     (int) sample rate to use for resulting song
    T      :     (int) time in seconds for dutation of each note in song
    n_pixels:     (int) how many pixels to use to make song
  Returns:
    song   :     (array) Numpy array of frequencies. Can be played by ipd.Audio(song, rate = sr)
  """
  # Convert image to HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Get shape of image
  height, width, depth = img.shape

  i = 0; j = 0 ; k = 0
  # Initialize array that will contain Hues for every pixel in image
  hues = []
  if random_pixels == False:
    for val in range(n_pixels):
      hue = abs(hsv[i][j][0]) # This is the hue value at pixel coordinate (i,j)
      hues.append(hue)
      i += 1
      j += 1

  else:
    for val in range(n_pixels):
      i = random.randint(0, height-1)
      j = random.randint(0, width-1)
      hue = abs(hsv[i][j][0])
      hues.append(hue)

  # Make dataframe containing hues and frequencies
  pixels_df = pd.DataFrame(hues, columns=['hues'])
  pixels_df['frequencies'] = pixels_df.apply(lambda row : hue2freq(row['hues'], scale), axis=1)
  frequencies = pixels_df['frequencies'].to_numpy()

  # Convert frequency to a note
  pixels_df['notes'] = pixels_df.apply(lambda row : librosa.hz_to_note(row['frequencies']), axis=1) # type: ignore

  # Convert note to a midi number
  pixels_df['midi_number'] = pixels_df.apply(lambda row : librosa.hz_to_midi(row['notes']), axis=1)

  # Make harmony disctionary
  # unison           = U0 ; semitone        = ST ; major second     = M2
  # minor third      = m3 ; major third     = M3 ; perfect forth    = P4
  # diatonic tritone = DT ; perfect fifth   = P5 ; minor sixth      = m6
  # major sixth      = M6 ; minor seventh   = m7 ; major seventh    = M7
  # octave           = O8
  harmony_select = {'U0': 1,
                    'ST': 16/15,
                    'M2': 9/8,
                    'm3': 6/5,
                    'M3': 5/4,
                    'P4': 4/3,
                    'DT': 45/32,
                    'P5': 3/2,
                    'm6': 8/5,
                    'M6': 5/3,
                    'm7': 9/5,
                    'M7': 15/8,
                    'O8': 2}
  
  harmony = np.array([]) # This array will contain the song harmony
  harmony_val = harmony_select[harmonize] # This will select the ration for the desired harmony

  song = np.array([]) # This array will contain the chosen frquencies in our song
  octaves = np.array([0.5, 1, 2]) # Go an octave below, same song or an octave higher
  t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable

  # Make a sing with numpy array
  for k in range(n_pixels):
    if use_octaves:
      octave = random.choice(octaves)
    else:
      octave = 1
    
    if random_pixels == False:
      val = octave * frequencies[k]
    else:
      val = octave * random.choice(frequencies)
    
    # Make note and hormony note
    note = 0.5*np.sin(2*np.pi*val*t)
    h_note = 0.5*np.sin(2*np.pi*harmony_val*val*t)

    # Place notes into corresponding arrays
    song = np.concatenate([song, note])
    harmony = np.concatenate([harmony, h_note])

  return song, pixels_df, harmony

