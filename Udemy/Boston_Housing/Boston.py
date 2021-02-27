from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()
df = pd.DataFrame(boston.data)

print(boston.feature_names)
print(df.isnull().values.any())
print(df.corr())

import matplotlib.pyplot as plt 
def plot_corr(df, size=12):
    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))#The subplots command in the background will call plt.figure(), and any keywords will be passed along
    ax.matshow(corr)   # heatmap with matshow, color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
plot_corr(df)

df.columns = boston.feature_names
df['House_Price'] = boston.target

#Saving data into X and y
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#Check R-square value on training data
print(linear_model.score(X_train, y_train))
#Check R-square value on test data
print(linear_model.score(X_test, y_test))

# Predicting the Test set results
y_predict = linear_model.predict(X_test)