import numpy as np

  #Remove columns in which there are too much NaN values (<80%)
def remove_col(x, nan_percentage = 0.8) :

  #Remove columns in which there are too much NaN values (<80%)
  to_delete = []
  for i in range(x.shape[1] - 1, 1, -1):
    num_NaN = np.count_nonzero(np.isnan(x[:,i]))
    p_NaN = num_NaN / x.shape[0]
    if p_NaN > nan_percentage:
      to_delete.append(i)

  x = x[:, [i for i in range(x.shape[1]) if i not in to_delete]]
  return x

def remove_outliers(data):

    filtered_data = data.copy()
    for i in range(filtered_data.shape[1]):
        col_data = filtered_data[:,i]
        q1 = np.percentile(col_data,25)
        q3 = np.percentile(col_data,75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr

        col_data[(col_data<lower_bound)|(col_data>upper_bound)] = np.median(col_data)
        filtered_data[:,i]= col_data
    return filtered_data
  
  def standardize(data):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis= 0)
    for i in range(len(std)):
        if std[i] < 1e-10:
            std[i] = 1
    standardized_data = (data - mean)/std
    return standardized_data

def remove_correlated_columns(data, correlation_threshold=0.1):
  uncorrelated_data = data.copy()
  upper_triangle = np.triu(np.ones((data.shape[1], data.shape[1]), dtype=bool), k=1)

  indices_to_delete = []

  for i in range(data.shape[1]):
    for j in range(i + 1, data.shape[1]):
      if upper_triangle[i, j]:
        corr = np.corrcoef(uncorrelated_data[:, i], uncorrelated_data[:, j])[0, 1]
        if np.abs(corr) >= correlation_threshold:
          indices_to_delete.append(j)

  uncorrelated_data = np.delete(uncorrelated_data, indices_to_delete, axis=1)

  return uncorrelated_data

def clean_data(data):
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
