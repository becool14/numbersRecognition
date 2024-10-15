def train_model(model, train_images, train_labels, epochs=10, batch_size=64):
    """Funkcja do trenowania modelu."""
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history
