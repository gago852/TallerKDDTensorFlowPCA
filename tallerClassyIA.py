# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#integrantes: GABRIEL GOMEZ, EDUARDO DE LA HOZ, STEPHANIA DE LA HOZ


# %%
import numpy as np
import pandas as pd
import scipy.linalg as la
from keras.models import Sequential
from keras.layers.core import Dense


# %%
def cleaner(df, k):
    return df[(abs(df[0]-np.mean(df[0])) <= k*np.std(df[0])) & (abs(df[1]-np.mean(df[1])) <= k*np.std(df[1])) & (abs(df[2]-np.mean(df[2])) <= k*np.std(df[2])) & (abs(df[3]-np.mean(df[3])) <= k*np.std(df[3]))]


def skynet(patrones, objetivos):
    patron = patrones.to_numpy()
    objetivo = objetivos.to_numpy()
    modelo = Sequential()
    modelo.add(Dense(10, input_dim=2, activation='sigmoid'))
    modelo.add(Dense(1, activation='relu'))
    modelo.compile(loss='mean_squared_error', optimizer='adam',
                   metrics=['categorical_accuracy'])
    modelo.fit(patron, objetivo, epochs=1000)
    puntaje = modelo.evaluate(patron, objetivo)
    print("\n%s: %.2f%%" % (modelo.metrics_names[1], puntaje[1]*100))
    return modelo


def prueba(modelo, dfPrueba):
    conjuntoPrueba = dfPrueba.to_numpy()
    preduccion = modelo.predict(conjuntoPrueba)
    return preduccion


# %%
df = pd.read_table('irisdata.txt', skiprows=9, header=None)
dfClean = df.copy()
cat = df.iloc[:, 4]
catn = cat.replace([1, 2], [0.5, 1])
catn = np.array(catn)
df = df.drop(columns=4)
rawdata = np.array(df)
covRawData = np.cov(rawdata.T)
resultRaw = la.eig(covRawData)
eugenVector = resultRaw[1]
eugenValors = resultRaw[0].real


# %%
promedio = np.mean(rawdata)
x = rawdata-promedio
proyeccion = eugenVector.T[:][:2].T
xPC = x.dot(proyeccion)
dfFlores = pd.DataFrame(data=xPC)
dfFlores[2] = catn


# %%
p_train = 0.80  # Porcentaje de train.

dfFlores['entrenamiento'] = np.random.uniform(0, 1, len(dfFlores)) <= p_train
train, test = dfFlores[dfFlores['entrenamiento'] ==
                       True], dfFlores[dfFlores['entrenamiento'] == False]
dfFlores = dfFlores.drop('entrenamiento', 1)


# %%
algo = 0.50
train['entrenamiento'] = np.random.uniform(0, 1, len(train)) <= algo
mitad1, mitad2 = train[train['entrenamiento'] ==
                       True], train[train['entrenamiento'] == False]

mitad1['entrenamiento'] = np.random.uniform(0, 1, len(mitad1)) <= algo
mitad2['entrenamiento'] = np.random.uniform(0, 1, len(mitad2)) <= algo

df1, df2, df3, df4 = mitad1[mitad1['entrenamiento'] == True], mitad1[mitad1['entrenamiento']
                                                                     == False], mitad2[mitad2['entrenamiento'] == True], mitad2[mitad2['entrenamiento'] == False]

df1 = df1.drop('entrenamiento', 1)
df2 = df2.drop('entrenamiento', 1)
df3 = df3.drop('entrenamiento', 1)
df4 = df4.drop('entrenamiento', 1)


# %%
entreno1 = df2.append([df3, df4])
entreno2 = df1.append([df3, df4])
entreno3 = df1.append([df2, df4])
entreno4 = df1.append([df2, df3])

modelo1 = skynet(entreno1.drop(2, 1), entreno1[2])
modelo2 = skynet(entreno2.drop(2, 1), entreno2[2])
modelo3 = skynet(entreno3.drop(2, 1), entreno3[2])
modelo4 = skynet(entreno4.drop(2, 1), entreno4[2])

prediccion1 = prueba(modelo1, df1.drop(2, 1))
prediccion2 = prueba(modelo2, df2.drop(2, 1))
prediccion3 = prueba(modelo3, df3.drop(2, 1))
prediccion4 = prueba(modelo4, df4.drop(2, 1))


# %%
train = train.drop('entrenamiento', 1)
test = test.drop('entrenamiento', 1)

modeloTotal = skynet(train.drop(2, 1), train[2])

prediccionTotal = prueba(modeloTotal, test.drop(2, 1))


# %%
print(prediccionTotal.round(1))
print(prediccion1.round(1))
print(prediccion2.round(1))
print(prediccion3.round(1))
print(prediccion4.round(1))
