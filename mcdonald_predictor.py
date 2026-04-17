import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import os
import logging
from typing import Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputValidationError(Exception):
    pass


class ModelTrainingError(Exception):
    pass


class PredictionError(Exception):
    pass


class McDonaldPredictor:
    NUMERICAL_FEATURES = ['age', 'income']
    CATEGORICAL_FEATURES = ['gender', 'visit_frequency', 'satisfaction_level']
    ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    TARGET = 'liked_mcdonalds'

    VISIT_FREQUENCY_ORDER = ['rarely', 'monthly', 'weekly', 'daily']
    SATISFACTION_ORDER = ['low', 'medium', 'high']

    def __init__(self, model_path: str = 'model.joblib', preprocessor_path: str = 'preprocessor.joblib'):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model: Optional[Any] = None
        self.preprocessor: Optional[Any] = None
        self.evaluation_results: Dict[str, Any] = {}
        self.feature_names: list = []

        if os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path):
            self._load_model()
        else:
            self.train_model()

    def _validate_input_data(self, data: pd.DataFrame, for_training: bool = True) -> None:
        if for_training:
            missing_features = [f for f in self.ALL_FEATURES if f not in data.columns]
            if missing_features:
                raise InputValidationError(f"Missing required features: {missing_features}")

            if self.TARGET not in data.columns:
                raise InputValidationError(f"Missing target column: {self.TARGET}")
        else:
            missing_features = [f for f in self.ALL_FEATURES if f not in data.columns]
            if missing_features:
                raise InputValidationError(f"Missing required features for prediction: {missing_features}")

        if 'age' in data.columns:
            invalid_age = data[(data['age'] < 0) | (data['age'] > 120)]
            if not invalid_age.empty:
                raise InputValidationError(f"Invalid age values detected. Age must be between 0 and 120.")

        if 'income' in data.columns:
            invalid_income = data[data['income'] < 0]
            if not invalid_income.empty:
                raise InputValidationError(f"Invalid income values detected. Income cannot be negative.")

        if 'gender' in data.columns:
            valid_genders = {'male', 'female'}
            invalid_gender = data[~data['gender'].isin(valid_genders)]
            if not invalid_gender.empty:
                raise InputValidationError(f"Invalid gender values. Must be one of: {valid_genders}")

        if 'visit_frequency' in data.columns:
            valid_visits = set(self.VISIT_FREQUENCY_ORDER)
            invalid_visit = data[~data['visit_frequency'].isin(valid_visits)]
            if not invalid_visit.empty:
                raise InputValidationError(f"Invalid visit_frequency values. Must be one of: {valid_visits}")

        if 'satisfaction_level' in data.columns:
            valid_satisfaction = set(self.SATISFACTION_ORDER)
            invalid_satisfaction = data[~data['satisfaction_level'].isin(valid_satisfaction)]
            if not invalid_satisfaction.empty:
                raise InputValidationError(f"Invalid satisfaction_level values. Must be one of: {valid_satisfaction}")

        if for_training and self.TARGET in data.columns:
            valid_targets = {0, 1}
            invalid_target = data[~data[self.TARGET].isin(valid_targets)]
            if not invalid_target.empty:
                raise InputValidationError(f"Invalid target values. Must be 0 or 1.")

    def _preprocess_data(self, data: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
        processed = data.copy()

        if for_training:
            processed = processed.dropna(subset=self.ALL_FEATURES + [self.TARGET])
            processed = processed[
                (processed['age'] >= 18) &
                (processed['age'] <= 100) &
                (processed['income'] >= 0)
            ].copy()

            if processed.empty:
                raise ModelTrainingError("No valid data remaining after preprocessing.")
        else:
            processed = processed.dropna(subset=self.ALL_FEATURES)
            if processed.empty:
                raise PredictionError("No valid data remaining after preprocessing for prediction.")

        return processed

    def _build_preprocessor(self) -> ColumnTransformer:
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.NUMERICAL_FEATURES),
                ('cat', categorical_transformer, self.CATEGORICAL_FEATURES)
            ])

        return preprocessor

    def _build_ensemble_model(self) -> VotingClassifier:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )

        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='liblinear'
        )

        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        )

        ensemble = VotingClassifier(
            estimators=[
                ('random_forest', rf),
                ('logistic_regression', lr),
                ('gradient_boosting', gb)
            ],
            voting='soft',
            weights=[2, 1, 2]
        )

        return ensemble

    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf.fit(X, y)

        feature_importance = dict(zip(self.feature_names, rf.feature_importances_))
        return dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    def train_model(self) -> None:
        try:
            logger.info("Loading and preprocessing data...")
            data = pd.read_csv('mcdonald_data.csv')

            data = data.rename(columns={
                'visit_frequency': 'visit_frequency',
                'satisfaction_level': 'satisfaction_level',
                'liked_mcdonalds': 'liked_mcdonalds'
            })

            logger.info("Validating input data...")
            self._validate_input_data(data, for_training=True)

            logger.info("Preprocessing data...")
            processed_data = self._preprocess_data(data, for_training=True)

            X = processed_data[self.ALL_FEATURES]
            y = processed_data[self.TARGET]

            logger.info("Splitting data into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            logger.info("Building preprocessor...")
            self.preprocessor = self._build_preprocessor()

            logger.info("Fitting preprocessor and transforming data...")
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)

            self.feature_names = self._get_feature_names()

            logger.info("Building ensemble model...")
            self.model = self._build_ensemble_model()

            logger.info("Training ensemble model...")
            self.model.fit(X_train_processed, y_train)

            logger.info("Evaluating model...")
            y_train_pred = self.model.predict(X_train_processed)
            y_test_pred = self.model.predict(X_test_processed)

            self.evaluation_results = {
                'train': {
                    'accuracy': float(accuracy_score(y_train, y_train_pred)),
                    'f1_score': float(f1_score(y_train, y_train_pred, average='weighted')),
                    'classification_report': classification_report(y_train, y_train_pred, output_dict=True)
                },
                'test': {
                    'accuracy': float(accuracy_score(y_test, y_test_pred)),
                    'f1_score': float(f1_score(y_test, y_test_pred, average='weighted')),
                    'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
                },
                'feature_importance': self._calculate_feature_importance(
                    np.vstack([X_train_processed, X_test_processed]),
                    np.concatenate([y_train, y_test])
                )
            }

            logger.info(f"Model evaluation complete:")
            logger.info(f"  Train Accuracy: {self.evaluation_results['train']['accuracy']:.4f}")
            logger.info(f"  Train F1-score: {self.evaluation_results['train']['f1_score']:.4f}")
            logger.info(f"  Test Accuracy: {self.evaluation_results['test']['accuracy']:.4f}")
            logger.info(f"  Test F1-score: {self.evaluation_results['test']['f1_score']:.4f}")

            logger.info("Saving model and preprocessor...")
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.preprocessor, self.preprocessor_path)

            logger.info("Model training completed successfully!")

        except InputValidationError as e:
            logger.error(f"Input validation error during training: {str(e)}")
            raise ModelTrainingError(f"Input validation failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise ModelTrainingError(f"Model training failed: {str(e)}") from e

    def _get_feature_names(self) -> list:
        if self.preprocessor is None:
            return []

        feature_names = []
        for name, trans, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                if hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                    cat_names = trans.named_steps['onehot'].get_feature_names_out(features)
                    feature_names.extend(cat_names)
                else:
                    feature_names.extend([f"{f}_{i}" for f in features for i in range(len(features))])

        return feature_names

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading existing model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            self.feature_names = self._get_feature_names()

            data = pd.read_csv('mcdonald_data.csv')
            data = data.rename(columns={
                'visit_frequency': 'visit_frequency',
                'satisfaction_level': 'satisfaction_level',
                'liked_mcdonalds': 'liked_mcdonalds'
            })

            processed_data = self._preprocess_data(data, for_training=True)
            X = processed_data[self.ALL_FEATURES]
            y = processed_data[self.TARGET]

            X_processed = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_processed)

            self.evaluation_results = {
                'overall': {
                    'accuracy': float(accuracy_score(y, y_pred)),
                    'f1_score': float(f1_score(y, y_pred, average='weighted')),
                    'classification_report': classification_report(y, y_pred, output_dict=True)
                },
                'feature_importance': self._calculate_feature_importance(X_processed, y)
            }

            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load existing model: {str(e)}. Training new model.")
            self.train_model()

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("Validating input for prediction...")
            df = pd.DataFrame([features])
            self._validate_input_data(df, for_training=False)

            logger.info("Preprocessing input data...")
            df = self._preprocess_data(df, for_training=False)

            if self.preprocessor is None or self.model is None:
                raise PredictionError("Model or preprocessor not initialized.")

            logger.info("Transforming input features...")
            X_processed = self.preprocessor.transform(df)

            logger.info("Making prediction...")
            prediction = self.model.predict(X_processed)[0]
            proba = self.model.predict_proba(X_processed)[0]
            confidence = float(max(proba))

            result = {
                'prediction': int(prediction),
                'prediction_label': '喜欢' if prediction == 1 else '不喜欢',
                'confidence': confidence,
                'confidence_percent': round(confidence * 100, 1)
            }

            logger.info(f"Prediction complete: {result}")
            return result

        except InputValidationError as e:
            logger.error(f"Input validation error during prediction: {str(e)}")
            raise PredictionError(f"Input validation failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}") from e

    def get_evaluation_results(self) -> Dict[str, Any]:
        return self.evaluation_results

    def get_feature_importance(self) -> Dict[str, float]:
        return self.evaluation_results.get('feature_importance', {})

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'Ensemble (Voting Classifier)',
            'base_estimators': ['RandomForest', 'LogisticRegression', 'GradientBoosting'],
            'voting': 'soft',
            'features': self.ALL_FEATURES,
            'numerical_features': self.NUMERICAL_FEATURES,
            'categorical_features': self.CATEGORICAL_FEATURES,
            'target': self.TARGET
        }
