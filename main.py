import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt


signal, sr = librosa.load(librosa.util.example('mozart'))

plt.figure(figsize=(20, 5))
librosa.display.waveshow(signal, sr=sr)
plt.title('Waveplot', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Amplitude', fontdict=dict(size=15))
plt.show()
