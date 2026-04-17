from flask import Flask, render_template, request, jsonify
from mcdonald_predictor import McDonaldPredictor, PredictionError
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    logger.info("Initializing McDonaldPredictor...")
    predictor = McDonaldPredictor()
    logger.info("McDonaldPredictor initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    predictor = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not initialized'
            }), 500

        model_info = predictor.get_model_info()
        return jsonify({
            'success': True,
            'data': model_info
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/evaluation', methods=['GET'])
def get_evaluation():
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not initialized'
            }), 500

        evaluation = predictor.get_evaluation_results()
        return jsonify({
            'success': True,
            'data': evaluation
        })
    except Exception as e:
        logger.error(f"Error getting evaluation results: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not initialized'
            }), 500

        feature_importance = predictor.get_feature_importance()
        return jsonify({
            'success': True,
            'data': feature_importance
        })
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not initialized'
            }), 500

        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No input data provided'
            }), 400

        required_fields = ['age', 'income', 'gender', 'visit_frequency', 'satisfaction_level']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        try:
            data['age'] = int(data['age'])
            data['income'] = float(data['income'])
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': 'Invalid data types: age must be integer, income must be number'
            }), 400

        result = predictor.predict_single(data)
        return jsonify({
            'success': True,
            'data': result
        })

    except PredictionError as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'status': 'healthy',
        'model_initialized': predictor is not None
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
