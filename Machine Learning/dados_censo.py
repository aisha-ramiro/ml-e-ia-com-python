# %%
#Importação de bibliotecas	
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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
