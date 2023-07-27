import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv("Book1.csv")


x = dataset[['TV']]
y = dataset['Sales']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(x_train, y_train)


print("Intercept: ", slr.intercept_)
print("Coefficient: ", slr.coef_)



y_pred_slr= slr.predict(x_test)

print("Prediction for test set: {}".format(y_pred_slr))


slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_slr})
slr_diff.head()


plt.scatter(x_test,y_test)
plt.plot(x_test, y_pred_slr, 'Red')
plt.show()


from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_slr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_slr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr))
print('R squared: {:.2f}'.format(slr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)