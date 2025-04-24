import plotly.express as px

dados = px.data.iris()

# Criar um gráfico de dispersão
fig = px.scatter(
    dados,
    x="sepal_width",
    y="sepal_length",
    color="species",
    title="Exemplo de Gráfico com Plotly"
)

fig.show()
