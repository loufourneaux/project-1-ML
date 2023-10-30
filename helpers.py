"""Some helper functions for project 1."""
import csv
import numpy as np
import os
from functions import sigmoid


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respective numpy arrays.
    Remember to put the 3 files in the same folder as the rest and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsampled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """  
    Generate a minibatch iterator for a dataset.
    
    This function takes as input two iterables (here the output desired values 'y' and the input data 'tx') and
    outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    
    Parameters:
    - y : array-like, shape = [n_samples]
        The output desired values.
    - tx : array-like, shape = [n_samples, n_features]
        The input data.
    - batch_size : int
        Size of each mini-batch.
    - num_batches : int, optional, default = 1
        Number of batches to return.
    - shuffle : bool, optional, default = True
        Whether to shuffle the data before splitting into batches.
    Yields:
    - minibatch_y : array-like, shape = [batch_size]
        Mini-batch of desired output values.
    - minibatch_tx : array-like, shape = [batch_size, n_features]
        Mini-batch of input data.
    """    
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def prediction(x_te, w, threshold=0.55):
    """
    Make predictions based on the input features and weights.
    
    This function takes as input the features for prediction and the weights of the model, 
    and it returns the predicted labels. The sigmoid function is applied to the dot product 
    of the input features and weights to get the probabilities. Then, a threshold is applied 
    to classify each probability into a class label.
    
    Parameters:
    - x_te : array-like, shape = [n_samples, n_features]
        Input features for prediction.
    - w : array-like, shape = [n_features]
        Weights of the model.
    - threshold : float, optional, default = 0.55
        Classification threshold. Probabilities above this threshold will be classified as 1, 
        and below or equal as -1.
        
    Returns:
    - array-like, shape = [n_samples]
        Predicted labels. Each element is either 1 or -1.
    """
    
    # Calculate the dot product of the input features and weights
    y_pred_prob = np.dot(x_te, w)
    
    # Apply the sigmoid function to convert the dot product results into probabilities
    y_pred_prob = sigmoid(y_pred_prob)
    
    # Classify the results into two classes (1 or -1) based on the specified threshold
    y_pred_prob[y_pred_prob > threshold] = 1  # Probabilities above the threshold are classified as 1
    y_pred_prob[y_pred_prob <= threshold] = -1  # Probabilities below or equal to the threshold are classified as -1
    
    return y_pred_prob  # Return the array of predicted labels



def compute_f1_score(y_true, y_pred):
    """
    Compute the F1 Score, also known as balanced F-score or F-measure.
    
    The F1 Score is the harmonic mean of precision and recall, where an F1 Score reaches its 
    best value at 1 (perfect precision and recall) and worst at 0.
    
    Parameters:
    - y_true : array-like, shape = [n_samples]
        True labels. Each element is either 1 or -1.
    - y_pred : array-like, shape = [n_samples]
        Predicted labels by the model. Each element is either 1 or -1.
        
    Returns:
    - float
        The F1 Score of the prediction.
    """
    
    # Calculate True Positives (tp): The cases which were predicted as positive and are actually positive
    tp = np.sum((y_pred == 1) & (y_pred == y_true))
    
    # Calculate False Positives (fp): The cases which were predicted as positive but are actually negative
    fp = np.sum((y_pred == 1) & (y_pred != y_true))
    
    # Calculate False Negatives (fn): The cases which were predicted as negative but are actually positive
    fn = np.sum((y_pred == -1) & (y_pred != y_true))
    
    # Calculate Precision: The ratio of correctly predicted positive observations to the total predicted positives
    precision = tp / (tp + fp)
    
    # Calculate Recall (Sensitivity): The ratio of correctly predicted positive observations to the all observations in actual class
    recall = tp / (tp + fn)
    
    # Calculate F1 Score: The weighted average of Precision and Recall
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score  # Return the F1 Score


def compute_accuracy(y_pred, y_true):
    """
    Compute the accuracy of a classification model.
    
    Parameters:
    - y_pred: Predicted labels
    - y_true: True labels
    
    Returns:
    - accuracy: Accuracy as a floating-point value between 0 and 1.
    """
    # Ensure that the input arrays have the same length
    if len(y_pred) != len(y_true):
        raise ValueError("Input arrays must have the same length.")

    # Count the number of correct predictions
    correct_predictions = sum(p == t for p, t in zip(y_pred, y_true))

    # Calculate the accuracy
    accuracy = correct_predictions / len(y_true)

    return accuracy

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.
    """
    # set seed
    np.random.seed(seed)

    idx = np.random.permutation(np.arange(len(x)))
    idx_max = np.floor(ratio * len(x)).astype(int)

    x_tr = x[idx][:idx_max]
    x_te = x[idx][idx_max:]
    y_tr = y[idx][:idx_max]
    y_te = y[idx][idx_max:]

    return x_tr, x_te, y_tr, y_te

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
