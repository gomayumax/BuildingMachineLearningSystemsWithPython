# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from pprint import pprint

# Whether to use Elastic nets (otherwise, ordinary linear regression is used)

# Load data:
data, target = load_svmlight_file('data/E2006.train')

# 7.3.1 データの中身を見てみよう
'''
print('Min target value: {}'. format(target.min()))
print('Max target value: {}'. format(target.max()))
print('Mean target value: {}'. format(target.mean()))
print('Std. dev. target: {}'. format(target.std()))
'''
# 7.3.1 普通の回帰をしてみよう
lr = LinearRegression()
lr.fit(data, target)
p = np.array(map(lr.predict,data))
p = p.ravel
e = p.astype(np.float)-target.astype(np.float)

total_sq_error = np.sum(e*e)
rmse_train = np.sqrt(total_sq_error/len(p))
print(rmse_train)

kf = KFold(len(data), n_folds=10)
error = 0
for training, test in kf:
    # 訓練
    lr.fit(data[training], target[training])
    # 回帰直線上の値
    p = lr.predict(data[test])
    # 実データとの距離
    e = p - data[test]
    error += np.sum(e * e)
rmse_10cv = np.sqrt(error / len(data))

print('RMSE 10-cross: {}'.format(rmse_10cv))


'''
lr = LinearRegression()

# Compute error on training data to demonstrate that we can obtain near perfect
# scores:

lr.fit(data, target)
pred = lr.predict(data)

print('RMSE on training, {:.2}'.format(np.sqrt(mean_squared_error(target, pred))))
print('R2 on training, {:.2}'.format(r2_score(target, pred)))
print('')

pred = np.zeros_like(target)
kf = KFold(len(target), n_folds=5)
for train, test in kf:
    lr.fit(data[train], target[train])
    pred[test] = lr.predict(data[test])

print('RMSE on testing (5 fold), {:.2}'.format(np.sqrt(mean_squared_error(target, pred))))
print('R2 on testing (5 fold), {:.2}'.format(r2_score(target, pred)))
'''
