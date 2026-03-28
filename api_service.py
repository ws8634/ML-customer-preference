from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Model service is running'})

@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        estimators = []
        for name, estimator in model.named_estimators_:
            estimators.append({
                'name': name,
                'type': type(estimator).__name__
            })
        
        return jsonify({
            'model_type': type(model).__name__,
            'voting_type': model.voting,
            'estimators': estimators,
            'n_classes': model.n_classes_
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
