import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)


warnings.filterwarnings("ignore")

path = r"Basics of Keras\KerasRegression\kc_house_data.csv"
df = pd.read_csv(path)
# print(df.head())

# EDA
# print(df.isnull().sum())
# print(df.describe().T)
# sns.displot(df['price'])
# sns.distplot(df['price'])
# sns.countplot(x='bedrooms', data=df)
# print(df.info())
df_heatmap = df.drop(["date"], axis=1)  # beacause date have
# print(df.drop(["date"], axis=1).corr()["price"].sort_values())
# sns.heatmap(df_heatmap.corr(), annot=True, cmap="Spectral", linewidths=5)
# sns.scatterplot(x="price", y="sqft_living", data=df)
# sns.boxplot(x='bedrooms', y='price', data = df)
# sns.scatterplot(x="price", y="long", data=df)
# sns.scatterplot(x="price", y="lat", data=df)
# sns.scatterplot(x="long", y="lat", data=df)
# sns.scatterplot(x="long", y="lat", hue="price", data=df)
# print(df.sort_values('price', ascending=False).head(20))
# print(len(df))
# print(len(df)*0.01)
non_top_1_perc = df.sort_values("price", ascending=False).iloc[216:, :]
# sns.scatterplot(x="long", y="lat", hue="price", data=non_top_1_perc)
# plt.show()
# sns.scatterplot(x="long", y="lat", hue="price", edgecolor=None, data=non_top_1_perc)
# plt.show()
# sns.scatterplot(x="long", y="lat", hue="price", edgecolor=None, alpha=0.2, data=non_top_1_perc)
# plt.show()
# sns.scatterplot(x="long", y="lat", hue="price", edgecolor=None, alpha=0.2, palette="RdYlGn", data=non_top_1_perc)
# sns.boxplot(x='waterfront', y='price', data=df)
# plt.show()

df.drop(["id"], axis=1, inplace=True)
df["date"] = pd.to_datetime(df["date"])  # convert into datetime object
# print(df["date"])
df["year"] = df["date"].apply(lambda date: date.year)
df["month"] = df["date"].apply(lambda date: date.month)
# print(df.head())
# lamda is same as below func
# def yera_extraction(date):
#     return date.year

# sns.boxplot(x='month', y='price', data=df)
# plt.show()

# print(df.groupby('month').mean()['price'])
# print(type(df.groupby('month').mean()['price']))
# df.groupby('month').mean()['price'].plot()
# df.groupby('year').mean()['price'].plot()
# plt.show()

df.drop(["date"], axis=1, inplace=True)
# print(df.head())

# print(df['zipcode'].value_counts())
df.drop(
    ["zipcode"], axis=1, inplace=True
)  # here we drop because of 70 categories in real if you want to you can categories make group of expensive zipcode and do lots more thing in feature engineering
print(df.head())
# print(df['yr_renovated'].value_counts()) # 0 means not renovated
# print(df['sqft_basement'].value_counts()) # 0 means there is no basement


# modeling
x = df.drop(
    ["price"], axis=1
).values  # .values returns numpy array and we have to convert it because tensorflow only teke numpy array
y = df[
    "price"
].values  # .values returns numpy array and we have to convert it because tensorflow only teke numpy array

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=101
)

# scaling (we do scaling on post split that way we only fit to the training set to prevent data leakage from the test set)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()
# print(x_train.shape) # here our trainnig dataset have 19 features so its good to have 19 neuron in our model
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(19, activation="relu"))
model.add(Dense(1))  # output

model.compile(optimizer="adam", loss="mse")
# hist = model.fit(x=x_train,y=y_train,validation_data=(x_test, y_test), batch_size=128, epochs=400)
# validation data means test data pass on model and after each epoch calculate how our model work and check loss on our test data but it not acctually affect on weight and biases of neural network
# batch size make batch of data it very typically to make batch size of power of 2 smaller batch size longer trainnig time and less likely to overfit your data

# model.save('KerasRegression.h5')

# loss_df = pd.DataFrame(hist.history)
# loss_df.plot()
# plt.show()
# loss_df.to_csv('Keras_loss_history.csv', index=False)

loss = pd.read_csv("Basics of Keras\KerasRegression\Keras_loss_history.csv")
# print(loss.head())
# loss.plot()
# plt.show()
# if val_loss(loss on test data) is decrease on increase on epoch so we can continue training without overfitting else you stop amd here val_loss and training loss same approx so here no overfitting

later_model = load_model("Basics of Keras\KerasRegression\KerasRegression.h5")

temp = np.array(later_model(x_test)).reshape(6480,) # if don't want to use predict method than you can use this but this give you tensor so you convert it to numpy array
# print(mean_squared_error(y_test, temp))
# print(np.sqrt(mean_squared_error(y_test, temp)))
# print(mean_absolute_error(y_test, temp))
# print(df['price'].describe())
# print(explained_variance_score(y_test, temp))


predictions = later_model.predict(x_test).reshape(6480,)
# print(prediction.shape)
dict = {"Actual": y_test, "Prediction": predictions}
# print(dict)
compare = pd.DataFrame(dict)
# print(compare.sample(10))
# compare.plot()
# plt.show()

# print(mean_squared_error(y_test, predictions))
# print(np.sqrt(mean_squared_error(y_test, predictions)))
# print(mean_absolute_error(y_test, predictions))
# print(df['price'].describe())
# print(explained_variance_score(y_test, predictions))
# we can retrain our model because we not reach at overfitting

# plt.scatter(y_test, predictions)
# plt.plot(y_test, y_test, 'r')
# plt.show()
# our work good for house price less than 2 or 3 million dollar but not work good for greater than that but 90+% house price between 2 or 3 million dollar

single_house = df.drop(["price"], axis=1).iloc[0]
single_house = scaler.transform(single_house.values.reshape(-1, 19)) # -1 means keeps those old dimensions alog axis
# print(single_house) 
# print(later_model.predict(single_house))
# print(df.head(1))

# we are kind of overshooting here beacause we are trying to fit extreme values (droping top values and retrain model to get better model)
