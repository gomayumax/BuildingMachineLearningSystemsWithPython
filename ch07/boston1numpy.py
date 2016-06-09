# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

# This script shows an example of simple (ordinary) linear regression

# lesson 7.1
import numpy as np
from sklearn.datasets import load_boston
import pylab as plt
from pprint import pprint

boston = load_boston()

## p142 部屋数と物件価格
"""
plt.scatter(boston.data[:,5], boston.target, color='r')
plt.xlabel('room num')
plt.ylabel('value')
plt.show()
"""
## p142 特徴量が１次元の時の回帰
## 特徴量配列を用意する
"""
x = boston.data[:, 5]
## 特徴配列xを二次元配列に変換(ただし、今回特徴量は1つなので二つ目の次元は1
x = np.array([[v] for v in x])
y = boston.target
## 回帰直線の傾きを取得
s, _, _, _ = np.linalg.lstsq(x, y)
y = s*x
## plot 
plt.scatter(boston.data[:,5], boston.target, color='r')
plt.plot(x, y, 'o-')
plt.xlabel('room num')
plt.ylabel('value')
plt.show()
"""

## p143 特徴量にバイアスを加えた回帰
"""
x = boston.data[:, 5]
## 特徴配列xを二次元配列に変換
x = np.array([[v,1] for v in x])
y = boston.target
## 回帰直線の"傾き"とバイアスを取得
(s,bias), _, _, _ = np.linalg.lstsq(x, y)
y = s*x+bias
## plot 
plt.scatter(boston.data[:,5], boston.target, color='r')
plt.plot(x, y, 'o-')
plt.xlabel('room num')
plt.ylabel('value')
plt.show()
"""

x = np.array([np.concatenate((v, [1])) for v in boston.data])
y = boston.target

# np.linal.lstsq implements least-squares linear regression
s, total_error, _, _ = np.linalg.lstsq(x, y)

rmse = np.sqrt(total_error[0] / len(x))
print('Residual: {}'.format(rmse))

# Plot the prediction versus real:
#plt.plot(np.dot(x, s), boston.target, 'ro')

# Plot a diagonal (for reference):
##plt.plot([0, 50], [0, 50], 'g-')
##plt.xlabel('predicted')
##plt.ylabel('real')
##plt.show()
