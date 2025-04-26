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


np.unique(base_credit['default'], return_counts=True) #Exibe os valores únicos da variável default
sns.countplot(x=base_credit['default']) #Exibe o gráfico de contagem da variável default
#plt.show() #Exibe o gráfico

plt.hist(x = base_credit['age'])
#plt.show()

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default') #Exibe o gráfico de dispersão   
grafico.show()