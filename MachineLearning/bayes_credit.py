#Importação de bibliotecas	
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import GaussianNB
import pickle

base_risco_credito = pd.read_csv('./MachineLearning/risco_credito.csv')

x_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values 

label_encoder_historia = LabelEncoder() #Transformando a variável categórica em numérica
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

x_risco_credito[:, 0] = label_encoder_historia.fit_transform(x_risco_credito[:, 0]) #Transformando a variável categórica em numérica
x_risco_credito[:, 1] = label_encoder_divida.fit_transform(x_risco_credito[:, 1])
x_risco_credito[:, 2] = label_encoder_garantia.fit_transform(x_risco_credito[:, 2])
x_risco_credito[:, 3] = label_encoder_renda.fit_transform(x_risco_credito[:, 3])

with open('./MachineLearning/risco_credito.pkl', 'wb') as file: #Salvando o modelo
    pickle.dump([x_risco_credito, y_risco_credito], file)

naive_risco_credito = GaussianNB() #Criando o modelo
naive_risco_credito.fit(x_risco_credito, y_risco_credito) #Treinando o modelo

previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
naive_risco_credito.classes_ #Verificando as classes do modelo
naive_risco_credito.class_prior_ #Verificando a probabilidade de cada classe
naive_risco_credito.count_ #Verificando a quantidade de cada classe