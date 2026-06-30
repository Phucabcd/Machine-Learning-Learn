import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer

data = pd.read_csv('missing_value.csv')
print(data)
print(type(data))

#conver data => dataframe to numpy array 
# conver = data.values 

#imputer xu li cac missing_values 
#strategy = mean, median, most_frequent, constant. read more ctrl + click strategy
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#fit doesn't change to data, just calculates the statistics
#fit => học & transform => áp dụng vào dữ liệu
result = imp.fit_transform(data)
# imp.fit(conver)
# result = imp.transform(conver)
print("Dữ liệu đã học fit & transform", imp.statistics_)

print(result)


