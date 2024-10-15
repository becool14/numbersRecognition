# MNIST Digit Classification

## Opis projektu
Projekt polega na stworzeniu sieci neuronowej, która klasyfikuje cyfry z zestawu danych MNIST. Wykorzystaliśmy TensorFlow i Keras do budowy modelu.

## Struktura projektu
- `main.py`: Główny plik uruchamiający proces trenowania i testowania modelu.
- `model.py`: Definicja architektury sieci neuronowej.
- `train.py`: Logika trenowania modelu.
- `evaluate.py`: Ocena modelu i obliczanie metryk.
- `utils.py`: Funkcje pomocnicze do ładowania i przygotowania danych.

## Jak uruchomić
1. Zainstaluj zależności:
    ```
    pip install -r requirements.txt
    ```
2. Uruchom program:
    ```
    python main.py
    ```

## Wyniki
Test accuracy: 98.4%

## Możliwości rozwoju
- Eksperymentowanie z innymi architekturami sieci neuronowej, np. CNN.
- Dodanie hiperparametryzacji i poszukiwanie najlepszych wartości.
