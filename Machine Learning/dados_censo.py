# %%
#Importação de bibliotecas	
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

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

