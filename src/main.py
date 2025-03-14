import pandas as pd
from azureml.core import Workspace, Experiment, Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar dados
file_path = 'vendas_sorvete.xlsx'
df = pd.read_excel(r'./data/vendas_sorvete.xlsx')

# Conectar ao workspace do Azure ML
ws = Workspace.from_config()

# Criar experimento
experiment = Experiment(workspace=ws, name='sorvete_regression')

# Separar dados em treino e teste
X = df[['Temperatura (°C)', 'Preço_Unitário (R$)']]
y = df['Quantidade_Vendas']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo de regressão
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsão
y_pred = modelo.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
print("Erro quadrático médio:", mse)

# Logar resultado no Azure ML
run = experiment.start_logging()
run.log("MSE", mse)
run.complete()