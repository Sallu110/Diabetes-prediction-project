# DIABETES PREDICTION PROJECT BY DEEP LEARNING  
In this project i have predicted diabetes on the basis various factors 

# implement neural net using keras sequential model 
# predict the onset of diabetes for the pima indians based on the available diagnostic data.

import pandas as pd 
import tensorflow as tf 
from keras.models import Sequential  
from keras.layers import Dense 

# read the file 

Data = pd.read_csv('diabetes.csv')
Data.isnull().sum(axis = 0)
X = Data.iloc[:,:-1]
Y = Data.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1234, stratify = Y)

# define the keras sequential model 

model = Sequential()
model.add(Dense(24,
                input_shape= (8,),
                activation= 'relu',
                kernel_initializer= 'RandomNormal'))



model.add(Dense(12,
                activation= 'relu',
                kernel_initializer= 'RandomNormal'))



model.add(Dense(1,
                activation= 'sigmoid',
                kernel_initializer= 'RandomNormal'))


# fit and train/ compile the model 

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train,Y_train,epochs = 160,batch_size = 10)

# evaluate the model 

accuracy_test = model.evaluate(X_test, Y_test)


y_pred_prob = model.predict(X_test)

predictions = (y_pred_prob > 0.5).astype('int32')
from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(Y_test,predictions)

# PROJECT END 

