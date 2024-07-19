
# Diabetes Prediction using Neural Network (Keras Sequential Model)
This project focuses on predicting the onset of diabetes for the Pima Indians based on available diagnostic data using a neural network implemented with Keras Sequential model.

# steps of project 
Introduction
Data Preprocessing
Model Training
Evaluation
Conclusion

# Introduction
In this project, we implement a neural network using the Keras Sequential model to predict whether a person is likely to develop diabetes based on diagnostic features such as glucose level, blood pressure, BMI, etc.

# Data Preprocessing
Import Libraries: Import pandas for data handling and TensorFlow for deep learning:

# import pandas as pd
# import tensorflow as tf
Read Data: Read the dataset and handle missing values if any:
Data = pd.read_csv('diabetes.csv')
Split Data: Split the dataset into features (X) and the target variable (Y):

X = Data.iloc[:, :-1]
Y = Data.iloc[:, -1]
Train-Test Split: Split the dataset into training and testing sets:

# from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234, stratify=Y)
# Model Training
Define the Model: Define a neural network model using Keras Sequential:

# from keras.models import Sequential
# from keras.layers import Dense

model = Sequential()
model.add(Dense(24, input_shape=(8,), activation='relu', kernel_initializer='RandomNormal'))
model.add(Dense(12, activation='relu', kernel_initializer='RandomNormal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='RandomNormal'))

Compile and Train the Model: Compile the model with appropriate loss function and optimizer, then train the model:
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=160, batch_size=10)

# Evaluation
Evaluate the Model: Evaluate the model's performance on the test set:

accuracy_test = model.evaluate(X_test, Y_test)
Confusion Matrix: Optionally, analyze model predictions using a confusion matrix for further insights:

![Screenshot 2024-07-18 171648](https://github.com/user-attachments/assets/faac5459-af75-45cd-adb5-b5288a29e895)

y_pred_prob = model.predict(X_test)
predictions = (y_pred_prob > 0.5).astype('int32')

![Screenshot 2024-07-18 171752](https://github.com/user-attachments/assets/a4c806ee-2a52-42bc-81a4-a973dddd645e)

![Screenshot 2024-07-18 171831](https://github.com/user-attachments/assets/1df1af58-4a2f-4be3-bd3e-c534589091c8)

![Screenshot 2024-07-18 171859](https://github.com/user-attachments/assets/49a1d76a-3373-429e-9958-c0504129d897)

![Screenshot 2024-07-18 171953](https://github.com/user-attachments/assets/cf3164cf-f129-4491-8870-272f9a9c31bb)



# from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predictions)

![Screenshot 2024-07-18 171237](https://github.com/user-attachments/assets/f6632f1f-a7da-4cab-9fd9-de311a45651e)


# Conclusion
This project demonstrates the implementation of a neural network using Keras Sequential model to predict diabetes onset based on diagnostic data of the Pima 
Indians. The model's performance is evaluated using accuracy metrics and confusion matrix analysis.












             
