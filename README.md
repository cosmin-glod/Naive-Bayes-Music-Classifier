# Music Genre Classification with Naive Bayes

A machine learning project that classifies music tracks into genres using Gaussian Naive Bayes classification with Principal Component Analysis (PCA) for dimensionality reduction.

## Overview

This project implements an intelligent music genre classification system that analyzes audio features to predict the genre of musical tracks. Using the popular GTZAN Dataset, the classifier can distinguish between 10 different music genres with high accuracy.

### Supported Genres
- Blues
- Classical  
- Country
- Disco
- Hip Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Quick Start

### Standard Classification (GTZAN Dataset)
Navigate to the project directory and run:
```bash
python main.py
```

### Custom Audio File Classification
The system will also prompt you to classify your own audio file! Simply provide the path to any `.wav` file when prompted, and the trained model will predict its genre.

## Architecture

### Dataset
- **Source**: GTZAN Dataset
- **Total tracks**: 999 audio files
- **Format**: WAV files (22050Hz, Mono, 16-bit)
- **Distribution**: ~100 tracks per genre (Jazz has 99 due to missing file #00054)

### Pipeline Overview
1. **Audio Feature Extraction** - Extract 61 audio characteristics
2. **Dimensionality Reduction** - Apply PCA to reduce to optimal feature count
3. **Classification** - Use Gaussian Naive Bayes for genre prediction

## Technical Implementation

### Audio Feature Extraction (61 Features)

#### Spectral Features
- **Chromatic Scale**: 12-class tonal representation (C to B with semitones)
- **Spectral Centroid**: Weighted average of frequencies (brightness measure)
- **Spectral Bandwidth**: Frequency range distribution
- **Spectral Rolloff**: Frequency containing 85% of spectral energy

#### Temporal Features
- **RMS Energy**: Root Mean Square amplitude measurement
- **Zero Crossing Rate**: Signal oscillation frequency
- **Tempo**: Beats per minute (BPM) estimation

#### Advanced Features
- **MFCC**: 20 Mel-Frequency Cepstral Coefficients (human auditory perception)
- **Harmonic/Percussive Components**: Separation of melodic vs rhythmic elements
- **Amplitude Envelope**: Maximum absolute amplitude per frame
- **Band Energy Ratio**: Low-frequency vs high-frequency energy balance (2kHz threshold)

### Principal Component Analysis (PCA)

Reduces the 61-dimensional feature space while preserving maximum variance:

```
K = argmin{k : Σ(i=1 to k) λᵢ / Σ(j=1 to N) λⱼ ≥ p}
```

Where:
- `λᵢ` = i-th eigenvalue in descending order
- `N` = total number of features (61)
- `p` = variance preservation ratio (optimal at 0.8)

**Result**: Typically reduces to ~13 principal components

### Gaussian Naive Bayes Classification

Uses Bayes' theorem with multivariate Gaussian distribution:

```
P(Cᵢ|x) = P(x|Cᵢ)P(Cᵢ) / P(x)
```

Where the likelihood follows:

```
P(x|Cᵢ) = 1/√((2π)ᵈ|Σ|) × exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
```

Parameters:
- `d` = number of features after PCA (~13)
- `μ` = mean vector for genre `Cᵢ`
- `Σ` = covariance matrix for genre `Cᵢ`
- `P(Cᵢ) = 1/10` (uniform prior)

## Project Structure

```
├── genres_original/          # GTZAN Dataset
├── GaussianBayes.py         # Naive Bayes classifier implementation
├── featureExtraction.py     # Audio feature extraction functions  
├── featureReduction.py      # PCA dimensionality reduction
├── createLabels.py          # Dataset label distribution
├── main.py                  # Main execution script
├── features.npy             # Extracted features cache
├── labels.npy               # Genre labels
└── pca.npy                  # PCA-transformed data
```

## Key Components

### `featureExtraction.py`
- **Libraries**: `librosa`, `numpy`, `os`
- **Functions**: Extract all 61 audio features with sliding window analysis
- **Performance**: ~20 minutes processing time (unoptimized STFT operations)

### `featureReduction.py` 
- **Method**: Eigenvalue decomposition of covariance matrix
- **Output**: Dimensionally reduced feature vectors maintaining 80% variance

### `GaussianBayes.py`
- **Implementation**: Multivariate Gaussian Naive Bayes
- **Features**: Log-probability computation, zero-value handling with epsilon
- **Libraries**: `numpy`, `scipy.stats.multivariate_normal`

### `main.py`
- **Data Split**: 80% training, 20% testing
- **Metrics**: Precision, Recall, F1-Score per genre + overall accuracy
- **Custom Classification**: Interactive feature to classify user-provided audio files
- **Workflow**: Train → Evaluate → Predict custom audio

## Performance Metrics

The system evaluates performance using:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)  
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Overall Accuracy**: Correct predictions / Total predictions

Results are displayed per genre and as aggregate statistics.

## Dependencies

- `numpy` - Numerical computing and matrix operations
- `librosa` - Audio analysis and feature extraction
- `scipy` - Statistical functions (multivariate normal distribution)
- `os` - File system navigation
- `random` - Data splitting

## Usage Examples

### Batch Classification & Model Training
```python
# Automatically runs on GTZAN dataset with 80:20 train/test split
python main.py
```

### Custom Audio File Classification
When you run `main.py`, the system will:
1. Train the model on the GTZAN dataset
2. Display performance metrics
3. **Prompt you to enter a path to your own audio file**
4. Extract features from your custom audio file
5. Apply the same PCA transformation
6. Predict the genre using the trained Gaussian Naive Bayes model

**Example workflow:**
```
Enter path to your audio file: /path/to/your/song.wav
Processing your audio file...
Extracting 61 audio features...
Applying PCA transformation...
Predicted Genre: Rock
```

## References

1. **Fauci, A., Cast, J., & Schulze, R. (2013)**. Music Genre Classification. *cs229.stanford.edu*

2. **Tzanetakis, G., & Cook, P. (2002)**. Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing, 10(5), 293-302.*

3. **Mohan, S., et al. (2018)**. Music Genre Classification using Machine Learning Techniques. *International Journal of Computer Trends and Technology (IJCTT) – Volume 62.*

4. **Valerio Velardo - The Sound of AI (2021)**. The Math Behind Music Genre Classification. *YouTube*

5. **Victor Lavrenko**. Principal Component Analysis. *YouTube*
