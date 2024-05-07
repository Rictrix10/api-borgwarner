import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Carregar os conjuntos de dados
train_data = pd.read_csv('../../datasets/TrainPoppetsDateTime.csv')

# Preparar os dados
X_train = train_data.drop(columns=['pPoppetpos', 'Ts'])
y_train = train_data['pPoppetpos']

# Treinar o modelo Gradient Boosted Trees
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Salvar o modelo treinado em um arquivo
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)
