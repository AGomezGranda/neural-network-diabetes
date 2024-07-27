from app.ml.model import NeuralNetwork
from app.ml.data_loader import load_and_preprocess_data
from app.core.config import settings


def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        'data/kaggle_dataset.csv')

    model = NeuralNetwork([X_train.shape[0], 10, 5, 1])
    model.train(X_train, y_train, iterations=1000, learning_rate=0.1)

    model.save_model(settings.MODEL_PATH)


if __name__ == "__main__":
    train_model()