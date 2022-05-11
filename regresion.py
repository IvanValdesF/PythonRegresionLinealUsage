import tensorflow as tf 
import numpy as np

import matplotlib.pyplot as plt

c = np.array([1,2,3,4,5,6,7,8],dtype=float)
f = np.array([58,67,79,93,107,122,138,155],dtype=float)

capa = tf.keras.layers.Dense(units=1,input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("empueza entrenamiento d modelo");
historial = modelo.fit(c,f,epochs=10000,verbose=False)



predict = modelo.predict([9])
print("La prediccion es: " + str(predict[0][0]))
resultado = capa.get_weights()
print("El valor de a es: " + str(resultado[1][0]))
print("El valor de b es: " +  str(resultado[0][0][0]))