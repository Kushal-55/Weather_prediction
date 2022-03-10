# Q1. Summary

In my project, I have implemented several major classification models on a rain prediction dataset and I have also used RNNs and Linear regression for stock analysis and prediction.

### Part 1

Objective - To predict whether it will rain or not in Australia the next day.

Accuracy Achieved - 86.57%

Methods Used -

* K Neighbors Classifier
* Logistic Regression
* Random Forest
* XG Boost Classifier
* Naive Bayes Classifier
* Ensemble learning
* Artificial Neural Networks

### Part 2

Objective - To learn about the RNNs and linear regression and to predict the closing price of a stock.

Validation loss - 0.010

Methods used -

* Linear regression
* RNN with LSTM


# Q2. Dataset Description

## Dataset Description

### For part 1

The dataset that I have chosen consists of 10 years of weather report from multiple locations across Australia.

* Number of columns(features)- 23
* Number of rows - 145460
* Total number of measurements - 2909200
* Target Variable - 'RainTomorrow'

This dataset consists of weather measurements made from 2008 -2017. It consists of several factors such as - Location, Humidity, Minimum and maximum temperatures, Rainfall, sunshine, pressure, evaporation and wind direction. The dataset consists a mixture of categorical and numerical variables, it is a classification dataset.

Each of this measurement is recorded everyday for different locations across Australia. It is defined as raining if the precipitation exceeds 1mm. 

For important factors affecting rain such as humidity, pressure and clouds, the measurements are taken in the morning(9am) and in the afternoon(3pm)

### For part 2 

In the second half of my project I used the data available for New york stock exchange. It consisted of the prices of all the stocks traded in the market from 2010 - 2016.

The dataset included -
Open - Opening price of a stock for that day
High - Highest price of a stock for that day
Low - Lowest price of a stock for that day
Close - The closing price of a stock for that day
Symbol - 562 unique symbols represented the companies which had their stocks listed on the market.
Date - The date and day of recording
Volume - The number of stocks of a particular company traded on that day

The dataset consisted of 851264 measurements and 7 features or columns.

## AUC Values

### Part 1 
|        Feature        |  AUC  |
|:----------------------|:-----:|
| Humidity3pm           | 0.790 |
| Sunshine              | 0.699 |
| Cloud3pm              | 0.696 |
| Rainfall              | 0.693 |
| Humidity9am           | 0.682 |
| Cloud9am              | 0.666 |
| Temp3pm               | 0.652 |
| Pressure9am           | 0.651 |
| WindGustSpeed         | 0.639 |
| Pressure3pm           | 0.638 |

### Part 2 

Since my dataset only contains 7 features -

|        Feature        |  AUC  |
|:----------------------|:-----:|
| High                  | 0.992 |
| Low                   | 0.982 |
| Open                  | 0.980 |
| Volume                 | 0.471 |

# Q3. Details

# Part 1

## Aim -

To determine whether it will rain or not tomorrow in Australia

## Approach -

## Preprocessing -

### Exploring the dataset -

The dataset consists of both numerical as well as categorical features. The categorical features have the datatype of 'object'. The numerical features have the datatype of 'float64'.

The dataset also consists of null values

'RainTomorrow' is our target variable since we have to predict this.

### Performing Univariate analysis 

I performed univariate analysis to find the unique values.

I changed the unique values of 'No', 'Yes' to '0','1' in the 'RainToday' and 'RainTomorrow' variables respectively.

Finding the Categorical variables-
'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday' and 'RainTomorrow'

## Feature Engineering -

### Finding null values and replacing them

I found all the null values in the numerical variables and replaced them with their mean.

I replaced all the null values of categorical variables with their most occuring value or mode.

Since our date value is an object, I created three columns with the Years, months and days, I removed the date column.

### Label Encoding Categorical variables
I used label encoding on categorical variables to convert each value to a numerical value for better processing.

### Discarding features
Discarding features which are not required such as 'WindGustDir' and others.

### Assigning X and y

My target variable y is 'Raintomorrow'
My X value is everything remaining except for y

## Implementing models

I used train_test_split rather than other cv techniques since it was taking too long to execute ANNs due to higher number of epochs.

## KNN

Implementing KNN classifier technique with K neighbours = 1, 5, 10.

In this technique I was able to achieve the highest accuracy for K= 10.

The KNN model works reasonably well with this dataset since it is a binary classification dataset, but performs even better at higher K values since this is a large dataset.

## Logistic Regression

I implemented the logistic regression classifier with the hyperparameters set to - max_iter = 1000 and solver = 'lbfgs'

I found these parameters optimal for this model. Since this model consists of high samples, maximum iterations were increased.
I chose lbfgs solver because it performs better than lib linear solver and it is also faster than other solvers on binary classification tasks.

## Random Forest

This model was used with the hyper parameter n_estimators = 100. This gave me optimal results since if I decreased the number of estimators, my model did not perform that well. This maybe due to the higher number of samples.

## XBG Classifier

The XGBoost classifier performed very well for my dataset. It is a model which had high computational accuracy and speed.

This was a new model that I studied and was surprised at the high amount of variability in the hyperparameters. I used the default penalty 'l1' since it performed better than 'l2' on my dataset. I tweaked some other hyperparameters- n_estimators = 100, max_depth = 10. Higher depth gave better results but increasing the depth above 10 overwhelmed the model and the performance decreased.

## Naive Bayes

This was also a model new to me and I had to go through a lot of documentation in order to get optimum results.

I implemented grid search cross validation along with the model.
I used variable smoothing parameter for better model performance
The Naive Bayes model performed quite well compared to other simpler models.

## Ensemble

I used the best performing methods so far to create an ensemble model. This included using XGB Classifier, Random forest model and knn with K =10.
I set the voting to hard and used the voting classifier to fit the data. Voting set to hard uses predicted class labels for the majority rule voting. The error rate was very low for the model but surprisingly not lower than some individual models.

## ANN
### ANN Model 1

I created an Artificial neural network to check its performance.
Here I created two models, one with dropout layers and one without.

In the first layer I created a basic sequential ANN model and added layers.

I also specified early stopping so that the model stops the training once the results have stopped improving.
The binary cross entropy loss worked best for my dataset and for the first two layers, I set the activation as 'relu' and for the last layer I set it to sigmoid.

The model performed well on the dataset with the second highest accuracy. The model was also able to predict well on testing data.

I have attached the accuracy vs epoch figures for both the neural networks.

### ANN Model 2

In the second ANN model, I used the same early stopping parameters as the first neural network but I made this network a lot more complex.

I set the kernel_ initializer to 'uniform'. This means that the layer is executed with uniformly distributed weights.

I added more layers to the model and added two dropout layers as a regularization method for the model.

This model performed well and gave a high accuracy.


## Conclusion

Even after using the more complex and highly intensive neural networks, the best performing model was the XG Boost Classifier. It provided the accuracy of 86.57% and the lowest error rate. 

Accuracy -
* KNN1 - 79.8%
* KNN5 - 83.61%
* KNN10 - 84.27%
* Logistic Regression - 84.29%
* Random Forest - 85.59%
* XGB Classifier - 86.57%
* Naive Bayes Classifier - 83.14%
* Ensemble - 85.79%
* ANN 1 - 84.19%
* ANN 2 - 84.22%

## References -

https://www.researchgate.net/publication/242579096_An_Introduction_to_Logistic_Regression_Analysis_and_Reporting
https://stats.stackexchange.com/questions/142873/how-to-determine-the-accuracy-of-regression-which-measure-should-be-used
https://www.mn.uio.no/fysikk/english/people/aca/ivarth/works/in9400_nn_hpo_nas_hovden_r2.pdf
https://www.tech-quantum.com/basic-hyperparameter-tuning-for-neural-networks/
https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf
https://towardsdatascience.com/xgboost-mathematics-explained-58262530904a


# Part 2

## Aim -

To learn about how linear regression, RNN and inturn the Long short term memory RNNS work and to implement them to predict the closing price of a stock.

## Approach -

### Data Cleaning-

My data consisted of all the companies listed on the stock market, hence I had to filter the company out in order to get the data for that company.

I chose the Apple inc stock traded as 'AAPL' on the New york stock market.


### Exploring the dataset -

The dataset consisted of the feature - symbol 'AAPL', open, low , high, close, date and of the volume of shares traded on the stock market for that day.

There were no null values in the dataset

The dataset consisted of two categorical variables - symbol and date with the datatype object.

Finding the trajectory -

I plotted the figure for the opening and closing data for the stock from 2010- 2016 to find out if it had an upward trajectory or a downward trajectory. The stock prices increased with time, hence it had an upward trajectory. I have attached a figure for reference.

### Data Extraction

I used the 'close' feature as my target variable to predict the closing price.

I found out the correlation between close price and the remaining variables and found out that symbol, volume and date features had very low correlation so I dropped these features.

Open and close prices had negative correlation whereas high and low had negative correlations.

## Linear Regression

For linear regression, I was able to achieve an accuracy of 99.3%. This was due to the fact that I had less number of variables to work with, for instance, for my target variable('close'), there were 1762 instances.

I was able to get a mean square error of 0.38

I used the default parameters for linear regression(since it gave me high accuracy) and did not want to overcomplicate the model by using ridge or lasso parameters.

## LSTM

I had to read a lot about RNN and LSTM, I have attached the references.

The LSTM model was the best performing model for stock prediction and it is used for time series forecasting.

I had to create timesteps for my data in order to make it ready for LSTM.

I used return sequences as true so that the model returns a sequence of predefined values and a hidden state output for each time step.

I added a dropout layer for better performance and set a dense layer as an output layer.

I have attached the validation loss vs number of epochs graph.
The validation loss was very low(0.010) as the test predictions were accurate.

## Conclusion

I studied and applied the linear regression and LSTM models which performed high on the test set.

Accuracy for Linear Regression- 99.3%
Accuracy for LSTM on testing data - 98.9%

## References 

I used these references to learn about lstms and how to implement them -

https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/
https://arxiv.org/abs/1909.09586
https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext




```python

```
