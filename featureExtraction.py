import os

import librosa, librosa.feature
import numpy as np

'''
    The mean gives you the average level of spectral across the audio sample.
    The variance or standard deviation gives you information about how much the frequency varies over time.
'''
def getSongs():
    filename = []
    songsFolder = 'genres_original'
    for genre in os.listdir(songsFolder):
        for song in os.listdir(songsFolder + '/' + genre):
            filename.append(songsFolder + '/' + genre + '/' + song)
    return filename

def chroma(data):
    computedChroma = librosa.feature.chroma_stft(y = data)

    '''
    Chroma computes the strength of each pitch
    '''

    mean = np.mean(computedChroma)
    var = np.var(computedChroma)

    return mean, var

def rootMeanSquareEnergy(data):
    rms = librosa.feature.rms(y = data)

    '''
    Root mean square of amplitude
    '''

    mean = np.mean(rms)
    var = np.var(rms)

    return mean, var

def spectralCentroidAndBandwidth(data):
    # spectral centroid
    centroid = librosa.feature.spectral_centroid(y = data)

    '''
    Average frequency of sound
    '''

    centroidMean = np.mean(centroid)
    centroidVar = np.var(centroid)

    # bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y = data)

    '''
    Range of frequencies where energy is distributed
    '''

    bandwidthMean = np.mean(bandwidth)
    bandwidthVar = np.var(bandwidth)

    return centroidMean, centroidVar, bandwidthMean, bandwidthVar


def rolloff(data):

    r_data = librosa.feature.spectral_rolloff(y = data)

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

def mel_frequency_cepstral_coef(data):
    mfcc_data = librosa.feature.mfcc(y = data)

    mfcc_means = np.mean(mfcc_data, axis=1)
    mfcc_var = np.var(mfcc_data, axis=1)

    return mfcc_means, mfcc_var

def compute_features():

    songs = getSongs()

    all_features = []

    for song in songs[0:]:
        song_features = np.empty(61)
        data = librosa.load(song)

        # Extract and assign various audio features to song_features

        # Chroma features
        song_features[0], song_features[1] = chroma(data)

        # Root Mean Square Energy
        song_features[2], song_features[3] = rootMeanSquareEnergy(data)

        # Spectral Centroid and Bandwidth
        song_features[4:8] = spectralCentroidAndBandwidth(data)

        # Spectral Rolloff
        song_features[8], song_features[9] = rolloff(data)

        # Zero Crossing Rate
        song_features[10], song_features[11] = zero_crossing_rate(data)

        # Harmonic and Percussive Components
        song_features[12:16] = decompose_harmonic_percussive(data)

        # Tempo
        song_features[16] = tempo(data)

        # Band Energy Ratio
        song_features[17], song_features[18] = band_energy_ratio(data)

        # Amplitude Envelope
        song_features[19], song_features[20] = amplitude_envelope(data)

        # Mel-Frequency Cepstral Coefficients (MFCC) - Alternating between mean and variance
        mfcc_means, mfcc_vars = mel_frequency_cepstral_coef(data)
        song_features[21:61:2] = mfcc_means
        song_features[22:62:2] = mfcc_vars

        all_features.append(song_features)