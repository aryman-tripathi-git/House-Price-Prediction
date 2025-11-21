import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

house_price_dataset = sklearn.datasets.fetch_california_housing()

house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

house_price_dataframe['Price'] = house_price_dataset.target

X = house_price_dataframe.drop(['Price'], axis=1)
Y = house_price_dataframe['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = XGBRegressor()
model.fit(X_train, Y_train)

test_data_prediction = model.predict(X_test)
score_3 = metrics.r2_score(Y_test, test_data_prediction)
score_4 = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("R squared value : ", score_3)
print("Mean Absolute Error : ", score_4)

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()