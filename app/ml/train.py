from model import NeuralNetwork
from data_loader import load_and_preprocess_data
import os

def train_model():
    # Cargar y preprocesar los datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/diabetes.csv')
    
    # Verificar si los datos se cargaron correctamente
    if X_train is None or X_test is None or y_train is None or y_test is None:
        print("Error al cargar los datos. Asegúrate de que el archivo exista y tenga el formato correcto.")
        return

    # Inicializar el modelo con la estructura deseada
    model = NeuralNetwork([X_train.shape[0], 10, 5, 1])
    
    # Entrenar el modelo
    model.train(X_train, y_train, iterations=5000, learning_rate=0.01)
    
    # Verificar que el directorio para guardar el modelo exista
    #os.makedirs(os.path.dirname('model.npy'), exist_ok=True)
    
    # Guardar el modelo entrenado
    model.save_model('model.npy')
    print("Modelo guardado correctamente en 'model.npy'.")

if __name__ == "__main__":
    train_model()
