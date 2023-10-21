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



  #standardize values ?? no bc not continuous variables 
  
