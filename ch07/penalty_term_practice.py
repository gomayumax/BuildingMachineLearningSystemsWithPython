from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold
import numpy as np
import pylab as plt
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score
from pprint import pprint

boston = load_boston()
x = boston.data[:, 5]
x = np.array([[v] for v in x])
y = boston.target

kf = KFold(len(x), n_folds=5)

# 通常最小二乗法
lr = LinearRegression(fit_intercept=True)
# 訓練
lr.fit(x, y)

# 訓練誤差の RMSE
# 回帰直線上の値
z = lr.predict(x)
lr_p = lr.predict(x)
r2_train = r2_score(y, lr_p)
p = np.zeros_like(y)
for train, test in kf:
  lr.fit(x[train], y[train])
  p[test] = lr.predict(x[test])
r2_cv = r2_score(y, p)
print('liner regression R2 on training: {}'.format(r2_train))
print('liner regression R2 on 5-fold CV: {}'.format(r2_cv))

# Elastic netのモデルを作る
en = ElasticNet(alpha=0.5)
en.fit(x,y)
en_p = en.predict(x)
r2_train = r2_score(y, en_p)

# Lassoのモデルを作る
la = Lasso(alpha=0.5)
la.fit(x,y)
la_p = en.predict(x)
r2_train = r2_score(y, la_p)
p = np.zeros_like(y)
for train, test in kf:
  la.fit(x[train], y[train])
  p[test] = la.predict(x[test])
r2_cv = r2_score(y, p)
print('Lasso R2 on training: {}'.format(r2_train))
print('Lasso R2 on 5-fold CV: {}'.format(r2_cv))

# Ridgeのモデルを作る
ri = Ridge(alpha=0.5)
ri.fit(x,y)
ri_p = en.predict(x)
r2_train = r2_score(y, ri_p)

plt.scatter(boston.data[:,5], boston.target, color='r')
plt.plot(x, z, 'o-')
#plt.plot(x, en_p, 'x-')
plt.plot(x, la_p, 'v-')
#plt.plot(x, ri_p, 's-')
plt.xlabel('room num')
plt.ylabel('value')
plt.show()

