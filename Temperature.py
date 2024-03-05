import tensorflow as tf
import numpy as np
import matplotlib.pyplot as ptl

celsius = np.array([-40, -10, 0, 8, 15, 22, 38, 60], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100, 140], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss = "mean_squared_error"
)

print("Comenzando entrenamiento")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado")

ptl.xlabel("# Epoca")
ptl.ylabel("Magnitud de pérdida")
ptl.plot(historial.history["loss"])
ptl.show()

print("Hagamos una predicción")
input_celsius = float(input("Ingresa una temperatura en Celsius: "))
resultado = modelo.predict(np.array([input_celsius]))
print("El resultado es " + str(resultado[0][0]) + " Fahrenheit")