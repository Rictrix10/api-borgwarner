from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Carregar o modelo treinado
with open('models/GradientBoostingRegressor/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fazer previsões
def make_prediction(data):
    year = data['year']
    month = data['month']
    day = data['day']
    hour = data['hour']
    minute = data['minute']
    second = data['second']
    
    # Criar um DataFrame com os dados de entrada
    input_data = pd.DataFrame({
        'year': [year],
        'month': [month],
        'day': [day],
        'hour': [hour],
        'minute': [minute],
        'second': [second]
    })
    
    # Fazer a previsão usando o modelo treinado
    prediction = model.predict(input_data)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    # Extrair os parâmetros do corpo da solicitação
    data = request.json

    # Fazer a previsão com o modelo treinado
    prediction = make_prediction(data)

    # Retorna a resposta em formato JSON
    return jsonify({'Prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
