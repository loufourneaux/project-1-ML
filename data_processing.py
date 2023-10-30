import numpy as np


# Function to remove columns with too many NaN values
def remove_col(x, nan_percentage=0.8):
    """
  Remove columns from the data where the percentage of NaN values 
  exceeds a specified threshold.
  
  Args:
  x (numpy.ndarray): Input data.
  nan_percentage (float): Threshold for removing columns.
  
  Returns:
  numpy.ndarray: Data with some columns removed.
  """
    to_delete = [i for i in range(x.shape[1]) if np.mean(np.isnan(x[:, i])) > nan_percentage]
    return np.delete(x, to_delete, axis=1)


# Function to remove outliers in the data
def remove_outliers(data):
    """
  Remove outliers from the data. Outliers are values that lie 
  below Q1-1.5*IQR or above Q3+1.5*IQR.
  
  Args:
  data (numpy.ndarray): Input data.
  
  Returns:
  numpy.ndarray: Data with outliers removed.
  """
    for i in range(data.shape[1]):
        q1, q3 = np.percentile(data[:, i], [25, 75])
        iqr = q3 - q1
        outlier_indices = np.where((data[:, i] < q1 - 1.5 * iqr) | (data[:, i] > q3 + 1.5 * iqr))
        data[outlier_indices, i] = np.median(data[:, i])
    return data


# Function to standardize the data
def standardize(data):
    """
  Standardize the data by removing the mean and scaling to unit variance. (z-score Normalization)
  
  Args:
  data (numpy.ndarray): Input data.
  
  Returns:
  numpy.ndarray: Standardized data.
  """
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    return (data - mean) / (std + 1e-10)


# Function to remove correlated columns in the data
def remove_correlated_columns(data, correlation_threshold=0.8):
    """
  Remove columns from the data that have a correlation higher 
  than a specified threshold.
  
  Args:
  data (numpy.ndarray): Input data.
  correlation_threshold (float): Threshold for removing columns.
  
  Returns:
  numpy.ndarray: Data with some columns removed.
  """
    corr_matrix = np.corrcoef(data, rowvar=False)
    to_delete = np.any(np.triu(corr_matrix, k=1) > correlation_threshold, axis=0)
    return np.delete(data, np.where(to_delete), axis=1)


def clean_data(data):
    """
  Fill NaN values in each column with the median of non-NaN values of the same column.
  """
    clean_datas = data.copy()

    for i in range(data.shape[1]):
        col = clean_datas[:, i]
        is_nan = np.isnan(col)

        if is_nan.any():
            valid_values = col[~is_nan]
            if valid_values.size > 0:
                median = np.median(valid_values)
                col[is_nan] = median

    return clean_datas


def remove_useless_col(data):
    """
  Remove specified columns from the data.
  """
    col_to_remove = [
        1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 21, 22, 24, 52, 53, 54, 60, 98,
        104, 105, 119, 120, 121, 122, 123, 124, 125, 126, 130, 131, 132, 133, 166,
        179, 181, 211, 212, 216, 217, 219, 220, 221, 222, 226, 227, 228, 229, 235,
        236, 237, 239, 240, 241, 244, 245, 246, 256, 286, 310, 311, 316, 317, 320
    ]

    return np.delete(data, col_to_remove, axis=1)


def remove_zero_variance_columns(data, threshold=1e-10):
    """
  Remove columns from the data where the variance is below a certain threshold.
  """
    variances = np.var(data, axis=0)
    non_zero_variance_indices = np.where(variances > threshold)[0]

    return data[:, non_zero_variance_indices]


def clean_all(data):
    """
  Apply all functions used to clean the dataset
  """
    data_to_compute = remove_useless_col(data)
    data_to_compute = remove_col(data_to_compute)
    data_to_compute = clean_data(data_to_compute)
    data_to_compute = remove_outliers(data_to_compute)
    data_to_compute = remove_zero_variance_columns(data_to_compute)
    data_to_compute = standardize(data_to_compute)
    data_to_compute = remove_correlated_columns(data_to_compute)
    return data_to_compute
