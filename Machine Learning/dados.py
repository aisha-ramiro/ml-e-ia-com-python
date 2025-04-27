#Importação de bibliotecas	
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


#Exploração de dados
base_credit= pd.read_csv('credit_data.csv')

#Visualização de dados
base_credit.head(10) #Exibe os primeiros registros
base_credit.tail(10) #Exibe os últimos registros

base_credit.describe() #Exibe estatísticas descritivas da base de dados


base_credit[base_credit['income'] >= 10000] #Exbe os registros com renda maior ou igual a 10000
base_credit[base_credit['loan'] <= 1000] #Exibe os registros com empréstimo menor ou igual a 1000

#Geração de gráficos
np.unique(base_credit['default'], return_counts=True) #Exibe os valores únicos da variável default
sns.countplot(x=base_credit['default']) #Exibe o gráfico de contagem da variável default
plt.show() #Exibe o gráfico

plt.hist(x = base_credit['age'])
plt.show()

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default') #Exibe o gráfico de dispersão   
grafico.show()

#Tratamento de valores inconsistentes
base_credit.loc[base_credit['age'] <0 ] #Localiza os registros com idade menor que 0

base_credit2 = base_credit.drop('age', axis = 1) #Apagar a coluna inteira(de todos da base de dados)

base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index) #Apagar valores inconsistentes com filtro

base_credit['age'].mean() #Calcular a média da idade
base_credit['age'][base_credit['age'] > 0].mean() #Calcular a média da idade apenas maior que 0
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92  #Substituir os valores inconsistentes pela média da idade

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default') #Exibe o gráfico de dispersão   
grafico.show()

#Tratamento de valores faltantes
base_credit.isnull() #Exibe os valores nulos da base de dados
base_credit.isnull().sum() #Exibe a soma dos valores nulos da base de dados
base_credit.loc(pd.isnull(base_credit['age'])) #Exibe os registros com valores nulos na coluna idade
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True) #Substitui os valores nulos pela média da idade
