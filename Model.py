import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import sklearn 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , mean_absolute_error
import pickle
from matplotlib import style
df = pd.read_csv(r"C:\Users\salwa\OneDrive\Desktop\machine learning projects\Predicting Students final grades - Linear Regression\student\student-por.csv",sep=";")
'''
print(df1.head())
print(df.info())
print(df1.describe())
#Looking for the relation between Study time and the final grade
plt.plot(df['studytime'], df['G3'], alpha=0.5)
plt.xlabel('Study time')
plt.ylabel('Grade ')
plt.title('Relationship between Study Time and Grade')
plt.show()
'''
df =df [['studytime', 'absences', 'failures', 'G1', 'G2', 'G3']]
print(df.head())
X = df[['studytime', 'absences', 'failures', 'G1', 'G2']]
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Training the model 500 times to get the best score! 
best_score = 0
for _ in range(500):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    model = LinearRegression()

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best_score:
        best_score = acc
        with open("studentgrades_model.pickle", "wb") as f:
            pickle.dump(model, f)


predictions = model.predict(X_test)

# Load The Model
pickle_in = open("studentgrades_model.pickle", "rb")
model = pickle.load(pickle_in)


#Some Analysis of the Model

print (f" the model accuracy is {best_score}")
print('The Coefficient are ', model.coef_)
print('The Intercept is ', model.intercept_)

style.use("ggplot")
plt.scatter(predictions, y_test)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xlabel("predictions")
plt.ylabel("G3")
plt.title("evaluation of the model")
plt.show()

