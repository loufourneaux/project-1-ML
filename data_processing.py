import numpy as np

def remove_nan_col(x, nan_percentage=0.8):
    to_delete = []
    for i in range(x.shape[1] - 1, 1, -1):
        num_NaN = np.count_nonzero(np.isnan(x[:, i]))
        p_NaN = num_NaN / x.shape[0]
        if p_NaN > nan_percentage:
            to_delete.append(i)

    x = x[:, [i for i in range(x.shape[1]) if i not in to_delete]]
    return x

def remove_outliers(data):
    filtered_data = data.copy()
    for i in range(filtered_data.shape[1]):
        col_data = filtered_data[:, i]
        q1 = np.percentile(col_data, 25)
        q3 = np.percentile(col_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        col_data[(col_data < lower_bound) | (col_data > upper_bound)] = np.median(col_data)
        filtered_data[:, i] = col_data
    return filtered_data

def standardize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for i in range(len(std)):
        if std[i] < 1e-10:
            std[i] = 1
    standardized_data = (data - mean) / std
    return standardized_data

def remove_correlated_columns(data, correlation_threshold=0.08):
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
    
def remove_remaining_nan(data, continuous_indices, discrete_indices):
    changed_nan_values = data.copy()
    print(changed_nan_values.shape)
    for i in discrete_indices:
        col = changed_nan_values[:, i]
        is_nan = np.isnan(col)
        if is_nan.any():
            col[is_nan] = 0
            
    for i in continuous_indices:
        col = changed_nan_values[:,i]
        is_nan = np.isnan(col)
        if is_nan.any():
            valid_values = col[~is_nan]
            if valid_values.size > 0:
                median = np.median(valid_values)
                col[is_nan] = median
    return changed_nan_values
    
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

def categorize_colv2(data):
    rows, cols = data.shape
    discrete_indices = []
    continuous_indices = []

    for col in range(cols):
        max_value = np.nanmax(data[:, col])

        if max_value <= 99:
            discrete_indices.append(col)

            # Si la valeur maximale est <= 9
            if max_value <= 9:
                for row in range(rows):
                    if data[row, col] in [7, 8, 9]:
                        data[row, col] = 0

            # Si la valeur maximale est entre 9 et 99
            else:
                for row in range(rows):
                    if data[row, col] in [77, 88, 99]:
                        data[row, col] = 0
        else:
            continuous_indices.append(col)
            median_value = np.nanmedian([val for val in data[:, col] if val not in [777, 7777, 777777, 888, 8888, 888888, 999, 9999, 999999]])
            median_value = round(median_value)  # Arrondir la médiane à l'entier le plus proche
            for row in range(rows):
                if data[row, col] in [777, 7777, 777777, 888, 8888, 888888, 999, 9999, 999999]:
                    data[row, col] = median_value

    return data, discrete_indices, continuous_indices
    
def calculate_r_squared(X):
    r_squared_values = []
    for i in range(X.shape[1]):
        x0 = X[:, i]
        x_remaining = np.delete(X, i, axis=1)

        # Coefficients de la régression linéaire
        coeffs = np.linalg.lstsq(x_remaining, x0, rcond=None)[0]

        # Prédiction
        prediction = x_remaining.dot(coeffs)

        # Calcul du R carré
        r_squared = 1 - np.var(x0 - prediction) / np.var(x0)
        r_squared_values.append(r_squared)

    return r_squared_values
    
def count_high_r_squared(r_squared_values, threshold=0.95):
    # Compter le nombre de valeurs de R² qui sont supérieures au seuil
    high_r_squared_count = sum(1 for r2 in r_squared_values if r2 >= threshold)

    return high_r_squared_count
    
def remove_highly_collinear_vars(data, r_squared_values, threshold=0.95):
    # Obtenir les indices des variables avec un R² supérieur au seuil
    high_collinearity_indices = [i for i, r2 in enumerate(r_squared_values) if r2 >= threshold]

    # Éliminer ces variables du dataset
    cleaned_data = np.delete(data, high_collinearity_indices, axis=1)

    return cleaned_data, high_collinearity_indices

def remove_useless_col(data):
    col_to_remove = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 18, 19, 21, 22, 24, 52, 53, 54, 60, 98, 104, 105, 119, 120, 121, 122, 123, 124, 125, 126, 130, 131, 132, 133, 166, 179, 181, 211, 212, 216, 217, 219, 220, 221, 222, 226, 227, 228, 229, 235, 236, 237, 239, 240, 241, 244, 245, 246, 256, 286, 310, 311, 316, 317, 320]
    return np.delete(data, col_to_remove, axis=1)

#Apply this function before standardization
def remove_zero_variance_columns(data, threshold=1e-10):
    variances = np.var(data, axis=0)
    non_zero_variance_indices = np.where(variances > threshold)[0]
    return data[:, non_zero_variance_indices]
    
def clean_all(data):
    data_to_compute = remove_useless_col(data)
    print(data_to_compute.shape)
    data_to_compute, discrete_col, continuous_col = categorize_colv2(data_to_compute)
    print(data_to_compute.shape)
    data_to_compute = remove__nan_col(data_to_compute)
    print(data_to_compute.shape)
    data_to_compute = remove_remaining_nan(data_to_compute, continuous_col, discrete_col)
    print(data_to_compute.shape)
    data_to_compute = remove_zero_variance_columns(data_to_compute)
    print(data_to_compute.shape) 
    data_to_compute = remove_outliers(data_to_compute)
    print(data_to_compute.shape)     
    data_to_compute = remove_correlated_columns(data_to_compute)
    print(data_to_compute.shape)
    data_to_compute = standardize(data_to_compute)
    print(data_to_compute.shape)
    return data_to_compute.copy()

def clean_all_no_high_r2(data):

    data_to_compute = remove_useless_col(data)
    print(data_to_compute.shape)
    data_to_compute, discrete_col, continuous_col = categorize_colv2(data_to_compute)
    print(data_to_compute.shape)
    data_to_compute = remove__nan_col(data_to_compute)
    print(data_to_compute.shape)
    data_to_compute = remove_remaining_nan(data_to_compute, continuous_col, discrete_col)
    print(data_to_compute.shape)
    print(data_to_compute.shape)
    data_to_compute = remove_zero_variance_columns(data_to_compute)
    print(data_to_compute.shape) 
    data_to_compute = remove_outliers(data_to_compute)   
    r_squared_values = calculate_r_squared(data_to_compute)
    high_r_squared_count = count_high_r_squared(r_squared_values)
    data_to_compute, removed_indices = remove_highly_collinear_vars(data_to_compute,r_squared_values)
    print(data_to_compute.shape)    
    data_to_compute = remove_correlated_columns(data_to_compute)
    print(data_to_compute.shape)
    data_to_compute = standardize(data_to_compute)
    print(data_to_compute.shape)
    return data_to_compute.copy()

