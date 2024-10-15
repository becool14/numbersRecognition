from sklearn.metrics import classification_report

def evaluate_model(model, test_images, test_labels):
    """Oceń model na zbiorze testowym i wypisz raport."""
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc}")

    # Dodatkowa analiza wyników
    predictions = model.predict(test_images)
    predictions_classes = predictions.argmax(axis=1)
    true_classes = test_labels.argmax(axis=1)

    print(classification_report(true_classes, predictions_classes))
