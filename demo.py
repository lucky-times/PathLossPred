import os
import numpy as np
import pykalman

from arg import parse_args

# args = parse_args()
# ignore_file = os.path.join(args.train_file,'labels', 'ignore.txt')
#
# a = np.loadtxt(ignore_file)
# print(a)


import numpy as np
from sklearn.linear_model import LinearRegression
from pykalman import KalmanFilter

# 生成示例数据
np.random.seed(0)
n_timesteps = 100
time = np.arange(n_timesteps)
true_values = np.sin(time * 0.1) + np.random.normal(0, 0.1, n_timesteps)
observations = true_values + np.random.normal(0, 0.1, n_timesteps)

# 训练机器学习模型（线性回归）
X_train = time[:-1].reshape(-1, 1)
y_train = observations[:-1]
model = LinearRegression()
model.fit(X_train, y_train)

# 使用机器学习模型进行预测
X_test = time.reshape(-1, 1)
predictions = model.predict(X_test)

# 初始化卡尔曼滤波器
initial_state_mean = predictions[0]
initial_state_covariance = 1.0
transition_matrix = 1.0
observation_matrix = 1.0
transition_covariance = 0.1
observation_covariance = 0.1

kf = KalmanFilter(
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    transition_covariance=transition_covariance,
    observation_covariance=observation_covariance
)

# 使用卡尔曼滤波器校正预测结果
state_means, state_covariances = kf.filter(predictions)

# 结果对比
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time, true_values, label='True values')
plt.plot(time, observations, label='Observations', linestyle='dotted')
plt.plot(time, predictions, label='ML Predictions', linestyle='dashed')
plt.plot(time, state_means, label='Kalman Filtered Predictions')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Machine Learning Predictions with Kalman Filter Correction')
plt.show()
print(1)



