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

# Load the data from the pickle file
path = r'C:\Users\sssso\OneDrive\Documents\Course Structure\Mindmove\data\emg_day_one.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

# Reshape the data and concatenate horizontally
reshaped_dictionary = {}
for key, value in data.items():
    reshaped_value = np.array(value).reshape((720, 320, 64))
    reshaped_value_horizontal = np.hstack(reshaped_value)
    reshaped_dictionary[key] = reshaped_value_horizontal
    print(f"Key: {key}, Value shape: {reshaped_value_horizontal.shape}")

# Define the filter parameters
low_freq = 20  # Lower cutoff frequency in Hz
high_freq = 500  # Upper cutoff frequency in Hz
sampling_rate = 2048  # Sampling rate of the EMG data in Hz

# Initialize a dictionary to store the feature matrices for each movement
feature_matrices = {}

# Extract features for each movement
for movement, data_array in reshaped_dictionary.items():
    # Extract features for each channel
    all_features_channel = []
    for channel_idx in range(320):
        channel_data = data_array[channel_idx, :]

        # Apply a bandpass filter
        b, a = signal.cheby2(4, 20, [low_freq / (sampling_rate / 2), high_freq / (sampling_rate / 2)], btype='band',
                             output='ba')
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
        frequency_ratio = np.sum(power_spectrum[(power_spectrum >= 10) & (power_spectrum <= 150)]) / np.sum(power_spectrum)
        # Compute Autoregressive (AR) modeling features(time-domain)
        ar_coeffs = signal.lfilter(b, a, filtered_data)[:8]  # Choose first 8 AR coefficients
        ar_features = np.hstack((ar_coeffs.real, ar_coeffs.imag))

        # Compute Wavelet transform features(time-freq domain)
        wavelet_coeffs = signal.cwt(filtered_data, signal.ricker, np.arange(1, 11))
        wavelet_features = np.mean(np.abs(wavelet_coeffs), axis=1)

        # Combine all features into a feature vector
        channel_features = [mean_absolute_value, rms, variance, zero_crossing_rate, waveform_length,slope_sign_change,
                            median_frequency, mean_frequency, frequency_ratio] +list(ar_features) + list(wavelet_features)
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

# Get the unique labels from all_labels
unique_labels = np.unique(all_labels)
unique_labels = np.array(unique_labels)

# Print the shape of unique_labels
print("Shape of unique_labels:", unique_labels.shape)


# Split the data into training, validation, and testing sets
x_train_val, x_test, y_train_val, y_test = train_test_split(all_features, all_labels, test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.0625, random_state=42)
num_sample_points = x_test.shape[0]
print("Number of sample points in x_test:", num_sample_points)

# Normalize the feature data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)
print (x_test_scaled.shape)
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
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
print(test_preds.shape)

# Predict labels for the validation data using the best model
val_preds = best_clf.predict(x_val_scaled)

# # Perform permutation importance
# # Define the feature names
# feature_names = [
#     'Mean Absolute Value', 'RMS', 'Variance', 'Zero Crossing Rate', 'Waveform Length',
#     'Slope Sign Change', 'Median Frequency', 'Mean Frequency', 'Frequency Ratio',
#     'AR Coefficients (Real Part 1)', 'AR Coefficients (Real Part 2)', 'AR Coefficients (Real Part 3)',
#     'AR Coefficients (Real Part 4)', 'AR Coefficients (Real Part 5)', 'AR Coefficients (Real Part 6)',
#     'AR Coefficients (Real Part 7)', 'AR Coefficients (Real Part 8)',
#     'AR Coefficients (Imaginary Part 1)', 'AR Coefficients (Imaginary Part 2)',
#     'AR Coefficients (Imaginary Part 3)', 'AR Coefficients (Imaginary Part 4)',
#     'AR Coefficients (Imaginary Part 5)', 'AR Coefficients (Imaginary Part 6)',
#     'AR Coefficients (Imaginary Part 7)', 'AR Coefficients (Imaginary Part 8)',
#     'Wavelet Transform Coefficients (Scale 1)', 'Wavelet Transform Coefficients (Scale 2)',
#     'Wavelet Transform Coefficients (Scale 3)', 'Wavelet Transform Coefficients (Scale 4)',
#     'Wavelet Transform Coefficients (Scale 5)', 'Wavelet Transform Coefficients (Scale 6)',
#     'Wavelet Transform Coefficients (Scale 7)', 'Wavelet Transform Coefficients (Scale 8)',
#     'Wavelet Transform Coefficients (Scale 9)', 'Wavelet Transform Coefficients (Scale 10)'
# ]
#
# # Perform permutation importance
# result = permutation_importance(best_clf, x_val_scaled, y_val, n_repeats=10, random_state=42)
#
# # Extract the feature importances
# importances = result.importances_mean
#
# # Get the indices of the calculated features
# time_domain_indices = [feature_names.index(feature_name) for feature_name in [
#     'Mean Absolute Value', 'RMS', 'Variance', 'Zero Crossing Rate', 'Waveform Length',
#     'Slope Sign Change'
# ]]
# frequency_domain_indices = [feature_names.index(feature_name) for feature_name in [
#     'Median Frequency', 'Mean Frequency', 'Frequency Ratio'
# ]]
# ar_coeff_indices = [feature_names.index(f'AR Coefficients (Real Part {i+1})') for i in range(8)]
# ar_coeff_imag_indices = [feature_names.index(f'AR Coefficients (Imaginary Part {i+1})') for i in range(8)]
# wavelet_coeff_indices = [feature_names.index(f'Wavelet Transform Coefficients (Scale {i+1})') for i in range(10)]
#
# # Combine the indices
# feature_indices = (
#     time_domain_indices + frequency_domain_indices + ar_coeff_indices + ar_coeff_imag_indices + wavelet_coeff_indices
# )
#
# # Extract the feature importances for the calculated features
# importances_calculated_features = importances[feature_indices]
#
# # Get the feature names for the calculated features
# feature_names_calculated_features = [feature_names[i] for i in feature_indices]
#
# # Convert importances to percentages
# total_importance = np.sum(importances)
# importances_calculated_features_percentage = (importances_calculated_features / total_importance) * 100
#
# # Plot the feature importances for the calculated features
# plt.figure(figsize=(20,18))
# sns.barplot(x=feature_names_calculated_features, y=importances_calculated_features_percentage, palette='Set2')
# plt.title('Permutation Importance - Calculated Features')
# plt.xlabel('Feature')
# plt.ylabel('Importance (%)')
# plt.xticks(rotation=90)
# plt.show()
#
#
# ###########
#
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     plt.figure(figsize=(8, 6))  # Increase figure size
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, ha='right')  # Adjust rotation and alignment
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), ha='center', va='center', color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# # Get the unique labels from y_test
# unique_labels = np.unique(y_test)
#
# # Calculate the confusion matrix
# cnf_matrix = confusion_matrix(y_test, test_preds)
#
# # Calculate TP, TN, FP, FN
# TP = np.diag(cnf_matrix)
# TN = np.sum(cnf_matrix) - (np.sum(cnf_matrix, axis=0) + np.sum(cnf_matrix, axis=1) - TP)
# FP = np.sum(cnf_matrix, axis=0) - TP
# FN = np.sum(cnf_matrix, axis=1) - TP
#
# # Create a dataframe for the TP, TN, FP, FN values
# data = {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
# df = pd.DataFrame(data, index=unique_labels)
#
# # Plot non-normalized confusion matrix and table
# fig, axes = plt.subplots(2, 2, figsize=(20,20 ))  # Increase figure size
#
# # Plot non-normalized confusion matrix
# axes[0, 0].set_title('Non-normalized Confusion matrix')
# sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, ax=axes[0, 0], cbar=False, xticklabels=unique_labels, yticklabels=unique_labels, annot_kws={"fontsize": 8})
# axes[0, 0].set_xlabel('Predicted label')
# axes[0, 0].set_ylabel('True label')
#
# # Plot table of TP, TN, FP, FN values for non-normalized confusion matrix
# axes[0, 1].set_title('TP, TN, FP, FN')
# axes[0, 1].axis('off')
# table = axes[0, 1].table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
#
# # Adjust table properties
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(0.5, 2.5)  # Scale the table
#
# # Plot normalized confusion matrix
# axes[1, 0].set_title('Normalized Confusion matrix')
# sns.heatmap(cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis], annot=True, fmt='.3f', cmap=plt.cm.Blues, ax=axes[1, 0], cbar=False, xticklabels=unique_labels, yticklabels=unique_labels, annot_kws={"fontsize": 8})
# axes[1, 0].set_xlabel('Predicted label')
# axes[1, 0].set_ylabel('True label')
#
# # Calculate normalized TP, TN, FP, FN
# TP_norm = TP / np.sum(cnf_matrix, axis=1)
# TN_norm = TN / np.sum(cnf_matrix, axis=1)
# FP_norm = FP / np.sum(cnf_matrix, axis=1)
# FN_norm = FN / np.sum(cnf_matrix, axis=1)
#
# # Round the normalized values to 3 decimal places
# TP_norm_rounded = np.round(TP_norm, 3)
# TN_norm_rounded = np.round(TN_norm, 3)
# FP_norm_rounded = np.round(FP_norm, 3)
# FN_norm_rounded = np.round(FN_norm, 3)
#
# # Create a dataframe for the normalized TP, TN, FP, FN values
# data_norm = {'TP': TP_norm_rounded, 'TN': TN_norm_rounded, 'FP': FP_norm_rounded, 'FN': FN_norm_rounded}
# df_norm = pd.DataFrame(data_norm, index=unique_labels)
#
# # Plot table of TP, TN, FP, FN values for normalized confusion matrix
# axes[1, 1].set_title('Normalized TP, TN, FP, FN')
# axes[1, 1].axis('off')
# table_norm = axes[1, 1].table(cellText=df_norm.values, colLabels=df_norm.columns, rowLabels=df_norm.index, cellLoc='center', loc='center')
#
# # Adjust table properties for normalized confusion matrix
# table_norm.auto_set_font_size(False)
# table_norm.set_fontsize(8)
# table_norm.scale(0.7, 2.5)  # Scale the table
#
# # Adjust subplot spacing
# plt.subplots_adjust(top=0.9, right=0.9, hspace=0.5, wspace=0.5)  # Adjust spacing
#
# plt.show()
#
#
# ## Calculate accuracy and F1 scores for each classifier
# train_accuracy = accuracy_score(y_train, train_preds)
# val_accuracy = accuracy_score(y_val, val_preds)
# test_accuracy = accuracy_score(y_test, test_preds)
#
# train_f1 = f1_score(y_train, train_preds, average='weighted')
# val_f1 = f1_score(y_val, val_preds, average='weighted')
# test_f1 = f1_score(y_test, test_preds, average='weighted')
#
# train_precision = precision_score(y_train, train_preds, average='weighted')
# val_precision = precision_score(y_val, val_preds, average='weighted')
# test_precision = precision_score(y_test, test_preds, average='weighted')
#
# train_recall = recall_score(y_train, train_preds, average='weighted')
# val_recall = recall_score(y_val, val_preds, average='weighted')
# test_recall = recall_score(y_test, test_preds, average='weighted')
#
#
#
# # Print accuracy, F1 score, precision, and recall
# print("Training Accuracy:", train_accuracy)
# print("Validation Accuracy:", val_accuracy)
# print("Testing Accuracy:", test_accuracy)
#
# print("Training F1 Score:", train_f1)
# print("Validation F1 Score:", val_f1)
# print("Testing F1 Score:", test_f1)
#
# print("Training Precision:", train_precision)
# print("Validation Precision:", val_precision)
# print("Testing Precision:", test_precision)
#
# print("Training Recall:", train_recall)
# print("Validation Recall:", val_recall)
# print("Testing Recall:", test_recall)
# print("Best Hyperparameters:", best_params)
#
# ####barplot of score metrics of model####
# # Calculate the evaluation scores
# scores = {
#     'Dataset': ['Training', 'Validation', 'Testing'],
#     'Accuracy': [train_accuracy, val_accuracy, test_accuracy],
#     'F1 Score': [train_f1, val_f1, test_f1],
#     'Precision': [train_precision, val_precision, test_precision],
#     'Recall': [train_recall, val_recall, test_recall]
# }
#
# # Convert the scores dictionary to a DataFrame
# scores_df = pd.DataFrame(scores)
#
# # Reshape the DataFrame for plotting
# scores_df = scores_df.melt('Dataset', var_name='Metric', value_name='Score')
#
# # Plot the scores
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Dataset', y='Score', hue='Metric', data=scores_df, palette='Set2')
# plt.title('Model Performance')
# plt.xlabel('Dataset')
# plt.ylabel('Score')
# plt.legend(title='Metric')
# plt.show()
# ####barplot of score metrics of model######
#
#
# ####Tabular form of Classification report for each class#####
# # Calculate precision, recall, and F1-score for each class
# report = classification_report(y_test, test_preds, labels=unique_labels, target_names=unique_labels, output_dict=True)
#
# # Initialize the table data
# table_data = []
#
# # Populate the table with the scores for each class
# for class_name in unique_labels:
#     precision = report[class_name]['precision']
#     recall = report[class_name]['recall']
#     f1_score = report[class_name]['f1-score']
#     accuracy = report['accuracy']  # Access accuracy from the classification report
#     table_data.append([class_name, precision, recall, f1_score, accuracy])
# # Convert the table data to a formatted string
# table_str = tabulate(table_data, headers=['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy'], tablefmt='github')
#
# # Plot the table as an image
# fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed
# ax.axis('off')
# ax.table(cellText=table_data, colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy'], cellLoc='center', loc='center')
# plt.savefig('classification_report.png')  # Save the table as an image
# plt.show()
# ####Tabular form of Classification report for each class#####
##############time axis################

#########time axis################

 #Calculate the duration in seconds
duration = 3.37
test_preds_sorted = np.sort(test_preds)
# Create the time axis
time_axis = np.linspace(0, duration, len(test_preds_sorted))

# Plot test_preds against time_axis
plt.plot(time_axis,test_preds_sorted)
plt.xlabel('Time')
plt.ylabel('Predicted Labels')
plt.title('Predicted Labels vs Time')
plt.show()
################################
