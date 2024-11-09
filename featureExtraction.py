import os
import librosa
import numpy as np

'''
    The mean gives you the average level of spectral across the audio sample.
    The variance or standard deviation gives you information about how much the frequency varies over time.
'''

def getSongs():
    filename = []
    songsFolder = 'genres_original'
    for genre in os.listdir(songsFolder):
        for song in os.listdir(songsFolder + '/' + genre)
            filename.append(songsFolder + '/' + genre + '/' + song)
    return filename

def chroma(data):
    computedChroma = librosa.feature.chroma_stft(data)

    '''
    Chroma computes the strength of each pitch
    '''

    mean = np.mean(computedChroma)
    var = np.var(computedChroma)

    return mean, var

def rootMeanSquareEnergy(data):
    rms = librosa.feature.rms(data)

    '''
    Root mean square of amplitude
    '''

    mean = np.mean(rms)
    var = np.var(rms)

    return mean, var

def spectralCentroidAndBandwidth(data):
    # spectral centroid
    centroid = librosa.feature.spectral_centroid(data)

    '''
    Average frequency of sound
    '''

    centroidMean = np.mean(centroid)
    centroidVar = np.var(centroid)

    # bendwidth
    bandwidth = librosa.feature.spectral_bandwidth(data)

    '''
    Range of frequencies where energy is distributed
    '''

    bandwidthMean = np.mean(bandwidth)
    bandwidthVar = np.var(bandwidth)

    return centroidMean, centroidVar, bandwidthMean, bandwidthVar


def rolloff(data):

    r_data = librosa.feature.spectral_rolloff(data)

    '''
    Frequency below which 85% of energy is considered
    '''

    r_mean = np.mean(r_data)
    r_var = np.var(r_data)

    return r_mean, r_var

def zero_crossing_rate(data):
    zcr = librosa.feature.zero_crossing_rate(data)

    '''
    How jerky a song is
    '''

    zcr_mean = np.mean(zcr)
    zcr_var = np.var(zcr)

    return zcr_mean, zcr_var

def decompose_harmonic_percussive(data):

    h_data = librosa.effects.harmonic(data)

    '''
    Harmonic sound
    '''

    h_mean = np.mean(h_data)
    h_var = np.var(h_data)

    p_data = librosa.effects.percussive(data)

    '''
    Percussive sound
    '''

    p_mean = np.mean(p_data)
    p_var = np.var(p_data)

    return h_mean, h_var, p_mean, p_var
