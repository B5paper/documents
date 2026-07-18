from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi * 2, num=1000)
sin = np.sin(x)
plt.plot(x, sin)
plt.show()

x_train, x_test, sin_train, sin_test = train_test_split(x, sin, test_size=0.25, shuffle=False)
print(x_train[-1], x_test[0])

# LinearRegression 的 fit() 只接收 2d 矩阵
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
sin_train = sin_train.reshape(-1, 1)
sin_test = sin_test.reshape(-1, 1)

lin_reg = LinearRegression()
lin_reg.fit(x_train, sin_train)
print(lin_reg.coef_)
print(lin_reg.intercept_)

sin_pred = lin_reg.predict(x_test)
plt.plot(x_train, sin_train, 'b')
plt.plot(x_test, sin_test, 'g')
plt.plot(x_test, sin_pred, 'r')
plt.show()

# 显然线性模型只能预测出一条直线，无法得到 sin 的曲线
