
from google.colab import drive
drive.mount('/content/drive')

# importing libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from keras.layers import Dropout
from keras import callbacks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


def part1():
    # loading the dataset
    dataset = pd.read_csv(r"/content/drive/MyDrive/project/data/weatherAUS.csv")

    # Categorical variables - The categorical variables are : ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
    # replacing no to 0 and yes to 1
    dataset['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    dataset['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

    # filling all the null values with mean of those values for numerical variables
    dataset['MinTemp'] = dataset['MinTemp'].fillna(dataset['MinTemp'].mean())
    dataset['MaxTemp'] = dataset['MinTemp'].fillna(dataset['MaxTemp'].mean())
    dataset['Rainfall'] = dataset['Rainfall'].fillna(dataset['Rainfall'].mean())
    dataset['Evaporation'] = dataset['Evaporation'].fillna(dataset['Evaporation'].mean())
    dataset['Sunshine'] = dataset['Sunshine'].fillna(dataset['Sunshine'].mean())
    dataset['WindGustSpeed'] = dataset['WindGustSpeed'].fillna(dataset['WindGustSpeed'].mean())
    dataset['WindSpeed9am'] = dataset['WindSpeed9am'].fillna(dataset['WindSpeed9am'].mean())
    dataset['WindSpeed3pm'] = dataset['WindSpeed3pm'].fillna(dataset['WindSpeed3pm'].mean())
    dataset['Humidity9am'] = dataset['Humidity9am'].fillna(dataset['Humidity9am'].mean())
    dataset['Humidity3pm'] = dataset['Humidity3pm'].fillna(dataset['Humidity3pm'].mean())
    dataset['Pressure9am'] = dataset['Pressure9am'].fillna(dataset['Pressure9am'].mean())
    dataset['Pressure3pm'] = dataset['Pressure3pm'].fillna(dataset['Pressure3pm'].mean())
    dataset['Cloud9am'] = dataset['Cloud9am'].fillna(dataset['Cloud9am'].mean())
    dataset['Cloud3pm'] = dataset['Cloud3pm'].fillna(dataset['Cloud3pm'].mean())
    dataset['Temp9am'] = dataset['Temp9am'].fillna(dataset['Temp9am'].mean())
    dataset['Temp3pm'] = dataset['Temp3pm'].fillna(dataset['Temp3pm'].mean())

    # replacing null values of important features with mode for categorical variables
    dataset['RainToday'] = dataset['RainToday'].fillna(dataset['RainToday'].mode()[0])
    dataset['RainTomorrow'] = dataset['RainTomorrow'].fillna(dataset['RainTomorrow'].mode()[0])

    dataset['WindDir9am'] = dataset['WindDir9am'].fillna(dataset['WindDir9am'].mode()[0])
    dataset['WindGustDir'] = dataset['WindGustDir'].fillna(dataset['WindGustDir'].mode()[0])
    dataset['WindDir3pm'] = dataset['WindDir3pm'].fillna(dataset['WindDir3pm'].mode()[0])

    dataset['Date'] = pd.to_datetime(dataset['Date'])
    # extract year, month, day from date

    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.month
    dataset['Day'] = dataset['Date'].dt.day

    # dataset.shape

    # removing date column
    dataset = dataset.iloc[:, 1:]

    # label encoding categorical variables
    label_encoding = preprocessing.LabelEncoder()
    dataset['Location'] = label_encoding.fit_transform(dataset['Location'])
    dataset['WindDir9am'] = label_encoding.fit_transform(dataset['WindDir9am'])
    dataset['WindDir3pm'] = label_encoding.fit_transform(dataset['WindDir3pm'])
    dataset['WindGustDir'] = label_encoding.fit_transform(dataset['WindGustDir'])

    dataset = dataset.drop(['Location', 'RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)

    X = dataset.drop(['RainTomorrow'], axis=1)

    y = dataset['RainTomorrow']

    # using train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Testing out different models
    # not using kfoldcv due to higher epochs in NN

    # KNN for K = 1,5,10
    # knn 1
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(X_train, y_train)
    y_pred1 = knn1.predict(X_test)
    score1 = accuracy_score(y_test, y_pred1)
    print('Error rate for KNN = 1', 1 - score1)

    # knn5
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(X_train, y_train)
    y_pred2 = knn5.predict(X_test)
    score2 = accuracy_score(y_test, y_pred2)
    print('Error rate for KNN = 5', 1 - score2)

    # KNN 10
    knn10 = KNeighborsClassifier(n_neighbors=10)
    knn10.fit(X_train, y_train)
    y_pred3 = knn10.predict(X_test)
    score3 = accuracy_score(y_test, y_pred3)
    print('Error rate for KNN = 10', 1 - score3)

    # using logistic regression with higher iterations
    logreg = LogisticRegression(max_iter=1000, solver='lbfgs')
    logreg.fit(X_train, y_train)
    y_pred4 = logreg.predict(X_test)
    score4 = accuracy_score(y_test, y_pred4)
    print('Error rate for Logistic Regression', 1 - score4)

    # using random forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred5 = rf.predict(X_test)
    score5 = accuracy_score(y_test, y_pred5)
    print('Error rate for RF', 1 - score5)

    # using XG Boost classifier
    xgb = XGBClassifier(n_estimators=100, max_depth=10)
    xgb.fit(X_train, y_train)
    y_pred6 = xgb.predict(X_test)
    score6 = accuracy_score(y_test, y_pred6)
    print('Error rate for XGBClassifier', 1 - score6)

    # using naive bayes
    # predefining grid
    nbgrid = {
        'var_smoothing': np.logspace(0, -9, num=100)
    }
    # using grid search to implement the grid
    bayes = GridSearchCV(estimator=GaussianNB(), param_grid=nbgrid, cv=10, n_jobs=-1)
    bayes.fit(X_train, y_train)
    y_pred7 = bayes.predict(X_test)
    score7 = accuracy_score(y_test, y_pred7)
    print('Error rate for Naive Bayes', 1 - score7)

    # creating an ensemble model with the best performing classifiers
    xgbx = XGBClassifier(n_estimators=100, max_depth=10)
    rfx = RandomForestClassifier(n_estimators=100)
    knnx = KNeighborsClassifier(n_neighbors=10)
    estimators = [
        ('knn', knnx), ('rf', rfx), ('xgb', xgbx)
    ]
    # setting voting to hard
    ensemble = VotingClassifier(voting='hard', estimators=estimators)
    ensemble.fit(X_train, y_train)
    score11 = ensemble.score(X_test, y_test)
    print('Error rate for Ensemble model', 1 - score11)

    # Basic ANN Model
    # Model without dropout layers
    model1 = Sequential()

    model1.add(Dense(64, activation='relu', input_dim=19))

    model1.add(Dense(32, activation='relu'))

    model1.add(Dense(1, activation='sigmoid'))

    # early stopping settings
    early_stopping = callbacks.EarlyStopping(patience=12, min_delta=0.001,
                                             restore_best_weights=True)
    # Compiling the model
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model

    history1 = model1.fit(X_train, y_train, epochs=50, batch_size=100,
                          validation_data=(X_test, y_test),
                          verbose=1)


    # plot accuracy
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.title('Model 1 Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # evaluate the model
    score8 = model1.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model1.metrics_names[1], score8[1] * 100))

    # ANN with dropout layers
    model2 = Sequential()

    # layers

    model2.add(Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim=19))

    model2.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))

    model2.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))

    model2.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    # adding a dropout layer
    model2.add(Dropout(0.25))
    model2.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))

    model2.add(Dropout(0.5))
    model2.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN

    model2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the ANN
    history2 = model2.fit(X_train, y_train, batch_size=64, epochs=50, callbacks=[early_stopping], validation_split=0.2)

    # print(history2.history.keys())

    plt.plot(history2.history['accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.title('Model 2 Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    score9 = model2.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model2.metrics_names[1], score9[1] * 100))



if __name__ == "__main__":
    #part1()
    part2()

