import scipy.io as sio  
import numpy as np
from sklearn.ensemble import RandomForestRegressor

data = []
data = sio.loadmat('RF_2.mat')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
y_train = np.reshape(y_train, [np.shape(y_train)[0]])
y_test = np.reshape(y_test, [np.shape(y_test)[0]])
print('input done')

mdl = RandomForestRegressor(n_estimators=50, max_depth=2, max_features='sqrt')
print('parameter done')

mdl.fit(x_train,y_train)
y_pre = mdl.predict(x_test)

sio.savemat('result1.mat',{'pre': y_pre,'true': y_test})