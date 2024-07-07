import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl

'''
Data getting and Selecting the open close e the high low data
'''
data = quandl.get("NSE/TATAGLOBAL")
data['Open - Close'] = data['Open'] - data['Close']
data['High - Low'] = data['High'] - data['Low']
data = data.dropna()
X = data[['Open - Close', 'High - Low']]
# print(X.head())

'''
Storing the data where the next closing value was grater than the last closing value,  as +1,
and store the data where it wasnt as -1
'''
Y = np.where(data['Close'].shift(-1)>data['Close'],1,-1)
# print(Y)


'''
Plotting the results
'''

plt.figure(figsize=(16,8))
plt.plot(data['Open - Close'])
plt.show()