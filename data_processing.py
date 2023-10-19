import numpy as np

def remove_col(x) :

  #Remove columns in which there are too much NaN values (<80%)

  for i in range(x.shape[1] - 1, 2, -1):
    num_NaN = np.count_nonzero(x[:, i] == -999.000)
    p_NaN = num_NaN / x.shape[0]
    if p_NaN > 0.8:
      x = np.delete(x, i, 1)
    return x



  #standardize values ?? no bc not continuous variables 
  