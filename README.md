# Multimodal Clinical Prediction Model

This repository contains code for a multimodal machine learning model that combines clinical measurements and text data from the MIMIC-III database to predict in-hospital mortality.

## Overview

The model uses both structured clinical measurements and unstructured text data to make predictions. It employs a BERT-based text encoder and a neural network for processing numerical measurements.

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

This code uses the MIMIC-III dataset, which requires credentialed access. After obtaining access:

1. Download the MIMIC-III dataset
2. Place the CSV files in a directory
3. Update the `data_path` variable in `data_processing.py` to your directory with the downloaded CSV files

## Usage

1. Process the data:
```bash
python data_processing.py
```

2. Train the model:
```bash
python train.py
```

3. Evaluate the model:
```bash
python evaluate.py
```

4. Run the modality dropout extension:
```bash
python modality_dropout.py
```

## Model Architecture

The model consists of:
- A measurement encoder for numerical data
- A BERT-based text encoder for clinical notes
- A classifier that combines both modalities

## Results

The model achieves competitive performance compared to the baseline, with the following metrics:
- AUC-ROC: ~0.85
- AUC-PR: ~0.49
