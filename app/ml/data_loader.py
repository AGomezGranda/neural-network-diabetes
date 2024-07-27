import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """
    Carga y preprocesa los datos desde un archivo CSV.

    Parameters:
    file_path (str): Ruta del archivo CSV.

    Returns:
    tuple: Datos preprocesados en la forma de (X_train, X_test, y_train, y_test)
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"El archivo {file_path} no se encontró.")
        return None, None, None, None
    except pd.errors.EmptyDataError:
        print("El archivo está vacío.")
        return None, None, None, None
    except pd.errors.ParserError:
        print("Error al parsear el archivo.")
        return None, None, None, None

    if 'Outcome' not in data.columns:
        print("El archivo no contiene una columna 'Outcome'.")
        return None, None, None, None

    # Separar características y etiquetas
    X = data.drop('Outcome', axis=1).values
    y = data['Outcome'].values

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Escalar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Transponer X para que las columnas sean las características y las filas los ejemplos
    X_train = X_train.T
    X_test = X_test.T

    # Asegurar que y es un vector fila
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    return X_train, X_test, y_train, y_test
