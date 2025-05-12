# %%
#Importação de bibliotecas	
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle


# %%
#Carregando a base de dados ----------------------
base_census = pd.read_csv('census.csv') # Carregando o dataset
descricao_geral = base_census.describe() # Descrição do dataset
nulos = base_census.isnull().sum() #Verifica se há valores nulos no dataset

# %%
#Visualizando a base de dados ----------------------
financeiro = np.unique(base_census['income'], return_counts=True) #Verifica a quantidade de valores únicos na variável escolhida

# %%
sns.countplot(x = base_census['income']) # Gráfico de barras para visualizar a variável escolhida
plt.show()

plt.hist(x = base_census['age']) # Gráfico de histograma para visualizar a variável escolhida 'age'
plt.show()
plt.hist(x = base_census['education-num']) 
plt.show()
plt.hist(x = base_census['hour-per-week'])
plt.show()

grafico_agrupado = px.treemap(base_census, path=['workclass', 'age']) # Gráfico de treemap para visualizar um agrupamento com a variável escolhida 'workclass'
grafico_agrupado.show()

grafico_paralelo = px.parallel_categories(base_census, dimensions=['occupation', 'relationship']) # Gráfico de categorias paralelas para visualizar a relação com a variável escolhida 'occupation'
grafico_paralelo.show()

#Divisores entre previsores e classes ----------------------
# %%
x_census = base_census.iloc[:, 0:14].values # Previsores
print(x_census  )
y_census = base_census.iloc[:, 14].values # Classes
print(y_census) 

# Atributos catégoricos - LabelEncoder ----------------------

# %%
label_encoder_workclass = LabelEncoder() # A variável é categórica
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1]) # Transformando a variável categórica em numérica
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])


# %%
# Atributos categóricos - OneHotEncoder ----------------------
onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough') # one hot encoder para transformar as variáveis categóricas em numéricas
x_census = onehotencoder_census.fit_transform(x_census).toarray() # Transformando as variáveis categóricas em numéricas
print(x_census[0]) # Visualizando os dados transformados

#Escalonamento dos valores ----------------------
# %%
scaler_census = StandardScaler() # Escalonando os dados
x_census = scaler_census.fit_transform(x_census) # Transformando os dados
print(x_census[0]) # Visualizando os dados transformados

#Divisão das bases de treino e teste ----------------------
x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_census, y_census, test_size=0.25, random_state=0) # Dividindo os dados em treino e teste
x_credit_treinamento.shape # Visualizando o tamanho dos dados de treino
x_credit_teste.shape # Visualizando o tamanho dos dados de teste

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census, y_census, test_size=0.15, random_state=0) # Dividindo os dados em treino e teste
x_census_treinamento.shape # Visualizando o tamanho dos dados de treino
x_census_teste.shape # Visualizando o tamanho dos dados de teste

#Salvar as bases de dados ----------------------
# %%
with open('credito.pkl', mode='wb') as f: # Salvando os dados em um arquivo pickle
    pickle.dump((x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste), f) # Salvando os dados em um arquivo pickle

with open('censo.pkl', mode='wb') as f: # Salvando os dados em um arquivo pickle
    pickle.dump((x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste), f) # Salvando os dados em um arquivo pickle