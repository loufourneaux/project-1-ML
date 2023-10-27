import numpy as np
import matplotlib.pyplot as plt
import sys 
import csv


sys.path.append('..')
from helpers import *
from costs import *
from implementations import *

#%load_ext autoreload
#%autoreload 2

# LOAD DATA

x_train, x_test, y_train, train_ids, test_ids = load_csv_data('dataset_to_release', subsample=True)


#PROCESS DATA
new_data=clean_all(x_train[1:])

#Split data
x_tr, x_te, y_tr, y_te = split_data(new_data, y_train, 0.8)

#intialize random variables
initial_w=np.ones([new_data.shape[1],1])
max_iter= 50
gamma = 0.1
lambda_= 6

#linear regression using gradient descent
w, loss=mean_squared_error_gd(y_tr, x_tr, initial_w, max_iter,gamma)
y_pred=np.dot(x_te,w)
score = compute_f1_score(y_te,y_pred)
print(score)

#linear regression using stochastic gradient descent
w1, loss1=mean_squared_error_sgd(y_tr, x_tr, initial_w, max_iter, gamma)
y_pred1=np.dot(x_te, w1)
score1= compute_f1_score(y_te, y_pred1)
print(score1)

#least squares
w2, loss2=least_squares(y_tr, x_tr)
y_pred2=np.dot(x_te, w2)
score2= compute_f1_score(y_te, y_pred2)

#ridge regression
w3, loss3=ridge_regression(y_tr, x_tr, lambda_)
y_pred3=np.dot(x_te, w3)
score3 = compute_f1_score(y_te,y_pred3)

#logistic regression
w4, loss4=logistic_regression(y_tr, x_tr, initial_w, max_iter,gamma)
y_pred4=np.dot(x_te,w4)
score4 = compute_f1_score(y_te,y_pred4)

#reg logistic regression
w5, loss5=logistic_regression(y_tr, x_tr, initial_w, max_iter,gamma)
y_pred5=np.dot(x_te,w5)
score5 = compute_f1_score(y_te,y_pred5)





