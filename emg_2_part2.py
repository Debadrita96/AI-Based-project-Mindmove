import numpy as np
import pickle
import pandas as pd
from scipy import signal
from scipy.fft import fft
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import itertools
from tabulate import tabulate



#### Testing Data ####
# Load the data from the pickle file
path = r'C:\Users\sssso\OneDrive\Documents\Course Structure\Mindmove\data\emg_day_two.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

# Reshape the data and concatenate horizontally
reshaped_dictionary = {}
for key, value in data.items():
    reshaped_value = np.array(value).reshape((298, 320, 64))
    reshaped_value_horizontal = np.hstack(reshaped_value)
    reshaped_dictionary[key] = reshaped_value_horizontal
    print(f"Key: {key}, Value shape: {reshaped_value_horizontal.shape}")



# Define the filter parameters
low_freq = 20  # Lower cutoff frequency in Hz
high_freq = 500  # Upper cutoff frequency in Hz
sampling_rate = 2048  # Sampling rate of the EMG data in Hz
nyquist_freq = sampling_rate / 2

# Initialize a dictionary to store the feature matrices for each movement
feature_matrices = {}

# Extract features for each movement
for movement, data_array in reshaped_dictionary.items():
    # Extract features for each channel
    all_features_channel = []
    for channel_idx in range(320):
        channel_data = data_array[channel_idx, :]

        # Calculate the Nyquist rate for the channel data
        nyquist_freq_channel = sampling_rate / 2

        # Calculate the normalized cutoff frequencies
        low_norm = low_freq / nyquist_freq_channel
        high_norm = high_freq / nyquist_freq_channel

        # Apply a bandpass filter
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        filtered_data = signal.lfilter(b, a, channel_data.flatten())



        # Compute time-domain features
        mean_absolute_value = np.mean(np.abs(filtered_data))
        rms = np.sqrt(np.mean(filtered_data ** 2))
        variance = np.var(filtered_data)
        zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(filtered_data)))) / (2 * len(filtered_data))
        waveform_length = np.sum(np.abs(np.diff(filtered_data)))
        slope_sign_change = np.sum(np.diff(np.sign(filtered_data)) != 0)
        # Compute frequency-domain features
        power_spectrum = np.abs(fft(filtered_data)) ** 2
        median_frequency = np.median(power_spectrum)
        mean_frequency = np.sum(power_spectrum * np.arange(len(filtered_data))) / np.sum(power_spectrum)
        frequency_ratio = np.sum(power_spectrum[(power_spectrum >= 10) & (power_spectrum <= 150)]) / np.sum(
            power_spectrum)

        # Compute Autoregressive (AR) modeling features (time-domain)
        ar_coeffs = signal.lfilter(b, a, filtered_data)[:8]  # Choose first 8 AR coefficients
        ar_features = np.hstack((ar_coeffs.real, ar_coeffs.imag))
        # Compute Wavelet transform features(time-freq domain)
        wavelet_coeffs = signal.cwt(filtered_data, signal.ricker, np.arange(1, 11))
        wavelet_features = np.mean(np.abs(wavelet_coeffs), axis=1)

        # Combine all features into a feature vector
        channel_features = [mean_absolute_value, rms, variance, zero_crossing_rate, waveform_length, slope_sign_change,
                            median_frequency, mean_frequency, frequency_ratio] + list(ar_features) + list(
            wavelet_features)
        all_features_channel.append(channel_features)

    # Convert the features into a numpy array
    feature_matrix = np.array(all_features_channel)
    # Store the feature matrix for the current movement
    feature_matrices[movement] = feature_matrix
# Display the number of features calculated for each movement
for movement, feature_matrix in feature_matrices.items():
    num_features = feature_matrix.shape[1]
    print(f"Movement: {movement}, Number of Features: {num_features}")


# Combine all the feature matrices and create corresponding labels
all_features = np.concatenate(list(feature_matrices.values()))
all_labels = np.concatenate([[movement] * feature_matrix.shape[0] for movement, feature_matrix in feature_matrices.items()])

x_test = all_features
y_test = all_labels

#### Training Data ####
# Load the data from the pickle file
path1 = r'C:\Users\sssso\OneDrive\Documents\Course Structure\Mindmove\data\emg_day_one.pkl'

with open(path1, 'rb') as f:
    data1 = pickle.load(f)

# Reshape the data and concatenate horizontally
reshaped_dictionary1 = {}
for key, value in data1.items():
    reshaped_value1 = np.array(value).reshape((720, 320, 64))
    reshaped_value_horizontal1 = np.hstack(reshaped_value1)
    reshaped_dictionary1[key] = reshaped_value_horizontal1
    print(f"Key: {key}, Value shape: {reshaped_value_horizontal1.shape}")

# ... Rest of the code to extract features ...
# Define the filter parameters
low_freq1 = 20  # Lower cutoff frequency in Hz
high_freq1 = 500  # Upper cutoff frequency in Hz
sampling_rate1 = 2048  # Sampling rate of the EMG data in Hz
nyquist_freq1 = sampling_rate1 / 2


# Initialize a dictionary to store the feature matrices for each movement
feature_matrices1 = {}

# Extract features for each movement
for movement, data_array in reshaped_dictionary1.items():
    # Extract features for each channel
    all_channel_features1 = []
    for channel_idx in range(320):
        channel_data1 = data_array[channel_idx, :]

        # Calculate the Nyquist rate for the channel data
        nyquist_freq_channel1 = sampling_rate1 / 2

        # Calculate the normalized cutoff frequencies
        low_norm1 = low_freq1 / nyquist_freq_channel1
        high_norm1 = high_freq1 / nyquist_freq_channel1

        # Apply a bandpass filter
        b1, a1 = signal.butter(4, [low_norm1, high_norm1], btype='band')
        filtered_data1 = signal.lfilter(b1, a1, channel_data1.flatten())

        # Compute time-domain features
        mean_absolute_value1 = np.mean(np.abs(filtered_data1))
        rms1 = np.sqrt(np.mean(filtered_data1** 2))
        variance1 = np.var(filtered_data1)
        zero_crossing_rate1 = np.sum(np.abs(np.diff(np.sign(filtered_data1)))) / (2 * len(filtered_data1))
        waveform_length1 = np.sum(np.abs(np.diff(filtered_data1)))
        slope_sign_change1 = np.sum(np.diff(np.sign(filtered_data1)) != 0)

        # Compute frequency-domain features
        power_spectrum1 = np.abs(fft(filtered_data1)) ** 2
        median_frequency1 = np.median(power_spectrum1)
        mean_frequency1 = np.sum(power_spectrum1 * np.arange(len(filtered_data1))) / np.sum(power_spectrum1)
        frequency_ratio1 = np.sum(power_spectrum1[(power_spectrum1 >= 10) & (power_spectrum1 <= 150)]) / np.sum(power_spectrum1)

        # Compute Autoregressive (AR) modeling features (time-domain)
        ar_coeffs1 = signal.lfilter(b, a, filtered_data1)[:8]  # Choose first 8 AR coefficients
        ar_features1 = np.hstack((ar_coeffs1.real, ar_coeffs1.imag))

        # Compute Wavelet transform features (time-freq domain)
        wavelet_coeffs1 = signal.cwt(filtered_data1, signal.ricker, np.arange(1, 11))
        wavelet_features1 = np.mean(np.abs(wavelet_coeffs1), axis=1)

        # Combine all features into a feature vector
        channel_features1 = [mean_absolute_value1, rms1, variance1, zero_crossing_rate1, waveform_length1, slope_sign_change1,
                            median_frequency1, mean_frequency1, frequency_ratio1] + list(ar_features1) + list(wavelet_features1)
        all_channel_features1.append(channel_features1)

    # Convert the features into a numpy array
    feature_matrix1 = np.array(all_channel_features1)
    # Store the feature matrix for the current movement
    feature_matrices1[movement] = feature_matrix1
# Display the number of features calculated for each movement
for movement, feature_matrix1 in feature_matrices1.items():
    num_features1 = feature_matrix1.shape[1]
    print(f"Movement: {movement}, Number of Features: {num_features1}")

# Combine all the feature matrices and create corresponding labels
all_features1 = np.concatenate(list(feature_matrices1.values()))
all_labels1 = np.concatenate([[movement] * feature_matrix1.shape[0] for movement, feature_matrix1 in feature_matrices1.items()])

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(all_features1, all_labels1, test_size=0.2, random_state=42)

## Preparing the model ####

# Scale the features using StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1],
    'gamma': ['scale', 'auto']
}

# Create and train the SVM classifier using GridSearchCV for hyperparameter tuning
clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=15)
clf.fit(np.concatenate((x_train_scaled, x_val_scaled)), np.concatenate((y_train, y_val)))

# Get the best hyperparameters
best_params = clf.best_params_

# Train the SVM classifier with the best hyperparameters on the training set
best_clf = SVC(kernel='rbf', **best_params)
best_clf.fit(x_train_scaled, y_train)

######################################################
# Predict labels for the training data using the best model
train_preds = best_clf.predict(x_train_scaled)

# Predict labels for the testing data using the best model
test_preds = best_clf.predict(x_test_scaled)

# Predict labels for the validation data using the best model
val_preds = best_clf.predict(x_val_scaled)

####################
# Calculate F1-score, precision, and recall for the training set
train_f1 = f1_score(y_train, train_preds, average='weighted')
train_precision = precision_score(y_train, train_preds, average='weighted', zero_division=0)
train_recall = recall_score(y_train, train_preds, average='weighted', zero_division=0)
print("Training Set - F1-score:", train_f1)
print("Training Set - Precision:", train_precision)
print("Training Set - Recall:", train_recall)

# Calculate F1-score, precision, and recall for the testing set
test_f1 = f1_score(y_test, test_preds, average='weighted')
test_precision = precision_score(y_test, test_preds, average='weighted', zero_division=0)
test_recall = recall_score(y_test, test_preds, average='weighted', zero_division=0)
print("Testing Set - F1-score:", test_f1)
print("Testing Set - Precision:", test_precision)
print("Testing Set - Recall:", test_recall)

# Calculate F1-score, precision, and recall for the validation set
val_f1 = f1_score(y_val, val_preds, average='weighted')
val_precision = precision_score(y_val, val_preds, average='weighted', zero_division=0)
val_recall = recall_score(y_val, val_preds, average='weighted', zero_division=0)
print("Validation Set - F1-score:", val_f1)
print("Validation Set - Precision:", val_precision)
print("Validation Set - Recall:", val_recall)

##########barplot###########
# Define the dataset labels
dataset_labels = ['Training', 'Validation', 'Testing']

# Define the metric labels
metric_labels = [ 'F1 Score', 'Precision', 'Recall']

# Define the metric values
metric_values = [
    [train_f1, val_f1, test_f1],
    [train_precision, val_precision, test_precision],
    [train_recall, val_recall, test_recall]
]

# Create a DataFrame from the metric values
performance_df = pd.DataFrame(metric_values, index=metric_labels, columns=dataset_labels)

# Plot the bar plot
plt.figure(figsize=(10, 6))
performance_df.plot(kind='bar', colormap='Set2')
plt.title('Model Performance')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(title='Dataset')
plt.show()
##############################################

