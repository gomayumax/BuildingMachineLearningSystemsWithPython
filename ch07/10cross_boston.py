import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold 
from pprint import pprint

boston = load_boston()
x = boston.data
x = np.array([np.concatenate((v, [1])) for v in boston.data])
y = boston.target

# 線形回帰する
lr = LinearRegression(fit_intercept=True)
# 訓練
lr.fit(x, y)

# 訓練誤差の RMSE
# 回帰直線上の値
p = lr.predict(x)
# 実データとの距離
error = p - y
# 平均二乗平方根誤差
total_error = np.sum(error * error)
rmse_train = np.sqrt(total_error / len(p))
print('RMSE: {}'.format(rmse_train))

# 10分割交差検証誤差の RMSE
kf = KFold(len(x), n_folds=10)
error = 0
for training, test in kf:
    # 訓練
    lr.fit(x[training], y[training])
    # 回帰直線上の値
    p = lr.predict(x[test])
    # 実データとの距離
    e = p - y[test]
    error += np.sum(e * e)
rmse_10cv = np.sqrt(error / len(x))

print('RMSE 10-cross: {}'.format(rmse_10cv))
