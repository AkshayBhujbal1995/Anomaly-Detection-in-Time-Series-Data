Here's the complete README content with everything included in one file for your project:

```markdown
# Anomaly Detection in Time Series Data using Autoencoder

This repository contains a Python implementation of anomaly detection in time series data using an autoencoder. The autoencoder is a neural network that learns to compress and reconstruct input data, making it suitable for detecting anomalies in data sequences. The implementation uses TensorFlow, Keras, and other Python libraries for data manipulation and model evaluation.

## Overview

Anomaly detection is essential for identifying unusual patterns in time series data, which could indicate potential issues, fraud, or opportunities. This project demonstrates how to detect anomalies in time series data using an unsupervised learning technique called autoencoders.

In this project, we focus on anomaly detection for time series data with the "ambient_temperature_system_failure.csv" dataset from the Numenta Anomaly Benchmark (NAB). The goal is to identify anomalous readings related to system failure using an autoencoder model.

## Key Features

- **Unsupervised Learning**: Uses autoencoders, a neural network, to learn typical patterns and identify anomalies without labeled data.
- **Anomaly Scoring**: Calculates reconstruction error as an anomaly score, where high reconstruction errors indicate anomalies.
- **Precision, Recall, and F1 Score**: Evaluates model performance using precision, recall, and F1 scores to measure the accuracy of anomaly detection.

## Dataset

The dataset used is **ambient_temperature_system_failure.csv** from the **Numenta Anomaly Benchmark (NAB)**, which contains time series data of ambient temperature readings from a system experiencing failure. You can access the dataset at the following link:
[ambient_temperature_system_failure.csv](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv)

## Libraries Used

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **matplotlib / seaborn**: Visualization.
- **scikit-learn**: Evaluation metrics (Precision, Recall, F1 Score).
- **tensorflow**: Building and training the autoencoder model.
- **keras**: High-level neural networks API for easy model development.

## Installation

To run this project, you'll need to install the following Python packages:

```bash
pip install pandas numpy tensorflow keras scikit-learn matplotlib seaborn
```

## Usage

### Step 1: Importing Libraries and Dataset

First, the necessary libraries are imported, and the dataset is loaded into a pandas DataFrame:

```python
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

data = pd.read_csv(
    'https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv')
```

### Step 2: Preprocessing the Data

The `timestamp` column is dropped, and the remaining data is converted to `float32` format for memory efficiency:

```python
data_values = data.drop('timestamp', axis=1).values
data_values = data_values.astype('float32')
data_converted = pd.DataFrame(data_values, columns=data.columns[1:])
data_converted.insert(0, 'timestamp', data['timestamp'])
data_converted = data_converted.dropna()
```

### Step 3: Building the Autoencoder Model

The autoencoder model is defined using Keras layers. The encoder compresses the input data, and the decoder reconstructs it:

```python
data_tensor = tf.convert_to_tensor(data_converted.drop('timestamp', axis=1).values, dtype=tf.float32)

input_dim = data_converted.shape[1] - 1
encoding_dim = 10

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='relu')(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
```

### Step 4: Training the Autoencoder

The model is compiled and trained to minimize the mean squared error (MSE) between the input data and the reconstructed data:

```python
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data_tensor, data_tensor, epochs=50, batch_size=32, shuffle=True)
```

### Step 5: Anomaly Detection

The reconstruction error is calculated, and an anomaly score is generated for each data point. High error values indicate anomalies:

```python
reconstructions = autoencoder.predict(data_tensor)
mse = tf.reduce_mean(tf.square(data_tensor - reconstructions), axis=1)
anomaly_scores = pd.Series(mse.numpy(), name='anomaly_scores')
anomaly_scores.index = data_converted.index
```

### Step 6: Evaluation (Optional)

To evaluate the performance of the anomaly detection model, you can calculate precision, recall, and F1 score using ground truth labels if available.

## Results

- **Precision, Recall, and F1 Score**: The model performance is evaluated by calculating the precision, recall, and F1 score of the detected anomalies.
- **Visualizations**: Various visualizations can be added to inspect anomaly detection results and model performance.

## Contributing

If you would like to contribute to this project, feel free to submit a pull request with your improvements or new ideas. Please ensure that your code follows the project's coding style and passes all tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
