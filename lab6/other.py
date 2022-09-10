import math
from cmath import nan
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def linearer(lowerLimit, upperLimit):
    upperLimit += 1
    out = []
    while (lowerLimit < upperLimit):
        out.append(lowerLimit)
        lowerLimit += 1
    return out 


df=pd.read_csv('ukraine-covid-cut.csv') #отримання даних

y = df.iloc[:,3] #3 колонка

y=[i for i in y if i != np.NaN and i != nan and i!=0]

length_y = len(y)
size = len(y)

forecast = int(math.ceil(length_y*0.6))

x = []
x = linearer(0, (length_y - 1))

print('Dimension of X', len(x))
print('Dimension of y', len(y))

x = pd.DataFrame(x)

yf=y[size-forecast:]
xf=x[size-forecast:]
y=y[:-forecast]
x=x[:-forecast]
length_y = len(y)

print('Dimension of X', len(x)) #-дані, які будуть прогнозуватися
print('Dimension of y', len(y))

#моделі
model = LinearRegression()
#model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1e-3)) #err
#model = LassoCV(eps=0.002, n_alphas=100, fit_intercept=True)
model.fit(x, y)

new_data = linearer(0, (forecast + length_y))

new_data = pd.DataFrame(new_data)
prediction=model.predict(new_data)
pred_df = pd.DataFrame({'pred': prediction})

poly_reg = PolynomialFeatures(degree = 9)
X_poly = poly_reg.fit_transform(xf)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, yf)

plt.title('models')
plt.xlabel('days')
plt.ylabel('Covid-19 cases')
plt.plot(xf, yf, color = 'orange')
plt.plot(xf, lin_reg_2.predict(poly_reg.fit_transform(xf)), color = 'purple', label = 'polynomial')
plt.plot(pred_df['pred'].shift(length_y), color='blue', label='linear')
plt.plot(df['new_cases'], color='red', label='data')
plt.legend(loc='best')
plt.show()