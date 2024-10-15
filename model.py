from tensorflow.keras import models, layers

def create_model():
    """Zdefiniuj i zwróć model sieci neuronowej (DNN)."""
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))  # Przekształca obraz 28x28 w wektor
    model.add(layers.Dense(128, activation='relu'))  # Pierwsza warstwa ukryta
    model.add(layers.BatchNormalization())  # Normalizacja Batch Normalization po warstwie Dense
    model.add(layers.Dropout(0.2))  # Dropout dla regularizacji
    model.add(layers.Dense(10, activation='softmax'))  # Warstwa wyjściowa z 10 neuronami
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
