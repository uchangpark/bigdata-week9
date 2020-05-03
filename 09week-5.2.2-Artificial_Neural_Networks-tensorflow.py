# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:19:31 2020
"""
# =============================================================================
# linear regression 예제
# =============================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

# 가상적인 데이터 생성 
X = data = np.linspace(1,2,200)	# 시작값=1, 종료값=2, 개수=200
y = X*4 + np.random.randn(200) * 0.3	# x를 4배로 하고 편차 0.3정도의 가우시안 잡음추가

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=1, activation='linear'))
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(X, y, batch_size=1, epochs=30)
predict = model.predict(data)

plt.plot(data, predict, 'b', data, y, 'k.') # 첫 번째 그래프는 파란색 마커로
plt.show()			      	      # 두 번째 그래프는 검정색 .으로 그린다.

# =============================================================================
# XOR 문제
# =============================================================================
import tensorflow as tf
# tensorflow 설치 => conda install tensorflow
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
 
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, input_dim=2, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,  activation='sigmoid'))
 
sgd = tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mean_squared_error', optimizer=sgd) 
model.fit(X, y, batch_size=1, epochs=1000) # epochs=10000으로 하면 정확
print(model.predict(X))
	