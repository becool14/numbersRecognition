from utils import load_data, preprocess_data
from model import create_model
from train import train_model
from evaluate import evaluate_model

def main():
    # Załaduj dane
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Przygotuj dane (normalizacja i one-hot encoding)
    train_images, train_labels, test_images, test_labels = preprocess_data(train_images, train_labels, test_images, test_labels)

    # Zbuduj model
    model = create_model()

    # Przeprowadź trening
    history = train_model(model, train_images, train_labels)

    # Oceń model
    evaluate_model(model, test_images, test_labels)

if __name__ == "__main__":
    main()
