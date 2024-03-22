import numpy as np
from matplotlib import pyplot as plt


# Sztuczne sieci neuronowe
# Przyjrzyj się etapom i różnym operacjom z algorytmu propagacji wstecznej.

# Etap A. Przygotowanie
# 1.	Zdefiniuj architekturę sieci ANN. Ten krok wymaga zdefiniowania węzłów wejściowych, węzłów wyjściowych,
#       liczby warstw ukrytych, liczby neuronów w każdej warstwie ukrytej, używanej funkcji aktywacji itd.
# 2.	Zainicjuj wagi w sieci ANN. Wagi w sieci ANN trzeba zainicjować jakąś wartością. Można zastosować tu różne podejścia.
#       Najważniejszą zasadą jest stałe dostosowywanie wag wraz z uczeniem się sieci ANN na podstawie obserwacji treningowych.

# Etap B. Propagacja w przód. Ten proces przebiega tak samo jak przy samym korzystaniu z sieci. Wykonywane są te same obliczenia.
# Jednak w trakcie uczenia sieci prognozowane dane wyjściowe są porównywane z rzeczywistą klasą każdej obserwacji ze zbioru
# treningowego.

# Etap C. Uczenie
# 1.	Oblicz koszt. Po propagacji w przód jako koszt przyjmowana jest różnica między prognozowanymi danymi wyjściowymi a
#       rzeczywistą klasą obserwacji ze zbioru treningowego. Koszt określa, jak dobrze sieć ANN radzi sobie z prognozowaniem
#       klas obserwacji.
# 2.	Zaktualizuj wagi w sieci ANN. Wagi w sieci ANN to jedyne wartości, jakie mogą być aktualizowane przez samą sieć.
#       Architektura i konfiguracja zdefiniowane na etapie A nie zmieniają się w procesie uczenia sieci. Wagi kodują
#       inteligencję sieci i mogą być zwiększane lub zmniejszane, co wpływa na „siłę” danych wejściowych.
# 3.	Zdefiniuj warunek zakończenia pracy. Proces uczenia nie może trwać w nieskończoność. Podobnie jak w wielu innych
#       algorytmach omawianych w tej książce trzeba ustalić sensowny warunek zakończenia pracy. Jeśli zbiór danych jest
#       duży, możesz zdecydować, że w procesie uczenia sieci ANN użyjesz 500 obserwacji ze zbioru treningowego i przeprowadzisz
#       1000 iteracji. Oznacza to, że 500 obserwacji zostanie 1000 razy przekazanych do sieci, a po każdej iteracji algorytm
#       dostosuje wagi.


# Skalowanie zbioru danych do wartości z przedziału od 0 do 1 metodą min-max.
def scale_dataset(dataset, feature_count, feature_min, feature_max):
    scaled_data = []
    for data in dataset:
        example = []
        for i in range(0, feature_count):
            example.append(scale_data_feature(data[i], feature_min[i], feature_max[i]))
        scaled_data.append(example)
    return np.array(scaled_data)


# Skalowanie cech ze zbioru danych do wartości z przedziału od 0 do 1 metodą min-max.
def scale_data_feature(data, feature_min, feature_max):
    return (data - feature_min) / (feature_max - feature_min)


# Jako funkcja aktywacji używana jest funkcja sigmoidalna.
# np.exp reprezentuje stałą matematyczną zwaną liczbą Eulera i równą w przybliżeniu 2.71828.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Pochodna funkcji sigmoidalnej.
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Klasa zawierająca mechanizmy sztucznej sieci neuronowej.
class NeuralNetwork:
    def __init__(self, features, labels, hidden_node_count):
        # Zapisanie cech jako danych wejściowych dla sieci neuronowej.
        self.input = features
        # Inicjowanie losowymi wartościami wag połączeń między warstwą wejścią a warstwą ukrytą.
        self.weights_input = np.random.rand(self.input.shape[1], hidden_node_count)
       # print(self.weights_input)
        # Inicjowanie wyjść węzłów ukrytych wartością None.
        self.hidden = None
        # Inicjowanie wag połączeń między warstwą ukrytą a warstwą wyjściową.
        self.weights_hidden = np.random.rand(hidden_node_count, 1)
       # print(self.weights_hidden)
        # Odwzorowanie oczekiwanych danych wyjściowych na etykiety.
        self.expected_output = labels
        # Inicjowanie wartości wyjściowych zerami.
        self.output = np.zeros(self.expected_output.shape)


    def add_example(self, features, label):
        np.append(self.input, features)
        np.append(self.expected_output, label)

    # Propagacja w przód - obliczanie sum ważonych i wartości funkcji aktywacji.
    def forward_propagation(self):
        hidden_weighted_sum = np.dot(self.input, self.weights_input)
        self.hidden = sigmoid(hidden_weighted_sum)
        output_weighted_sum = np.dot(self.hidden, self.weights_hidden)
        self.output = sigmoid(output_weighted_sum)

    # Propagacja wsteczna - obliczanie kosztu i aktualizowanie wag.
    def back_propagation(self):
        cost = self.expected_output - self.output
    #    print('RZECZYWISTA: ')
    #    print(self.expected_output)
    #    print('PROGNOZOWANA: ')
    #    print(self.output)
    #    print('KOSZT: ')
    #    print(cost)
    #    print('UKRYTA: ')
    #    print(self.hidden)
        weights_hidden_update = np.dot(self.hidden.T, (2 * cost * sigmoid_derivative(self.output)))
    #    print('AKTUALIZACJA WAG WARSTWY UKRYTEJ:')
    #    print(weights_hidden_update)
        weights_input_update = np.dot(self.input.T, (np.dot(2 * cost * sigmoid_derivative(self.output), self.weights_hidden.T) * sigmoid_derivative(self.hidden)))
    #    print('AKTUALIZACJA WAG WARSTWY WEJŚCIOWEJ:')
    #    print(weights_hidden_update)

        # Aktualizowanie wag na podstawie pochodnej (nachylenia) funkcji straty.
        self.weights_hidden += weights_hidden_update
    #    print('WAGI WARSTWY UKRYTEJ:')
    #    print(weights_hidden_update)

        self.weights_input += weights_input_update
    #    print('WAGI WARSTWY WEJŚCIOWEJ:')
    #    print(weights_hidden_update)

def MSE( targets, predictions):
        return ((targets - predictions) ** 2).mean()

def classification_error(targets, predictions):
        y_pred = np.where(predictions > 0.5, 1, 0)
        return (targets != y_pred).mean()
def run_neural_network(feature_data, label_data, feature_count,  hidden_node_count, epochs):
    loss_history = []
    classification_error_history = []
    weights1_history = []
    weights2_history = []
    # Skalowanie zbioru danych metodą min-max.
    # Inicjowanie sieci neuronowej przeskalowanymi danymi i węzłami warstwy ukrytej.
    nn = NeuralNetwork(feature_data, label_data, hidden_node_count)
    # Uczenie sztucznej sieci neuronowej przez wiele iteracji, używając tych samych danych treningowych.
    for epoch in range(epochs):
        nn.forward_propagation()
        loss_history.append(MSE(label_data,nn.output))
        nn.back_propagation()
        classification_error_history.append(classification_error(label_data,nn.output))
        weights1_history.append(nn.weights_input)

        weights2_history.append(nn.weights_hidden)

    # print('DANE WYJŚCIOWE: ')
  #  for r in nn.output:
  #      print(r)

    # print('WAGI WARSTWY WEJŚCIOWEJ: ')
  #  print(nn.weights_input)
    weights1_history.append(nn.weights_input)
 #   print('WAGI WARSTWY UKRYTEJ: ')
  #  print(nn.weights_hidden)


    return loss_history,classification_error_history,weights1_history,weights2_history

if __name__ == '__main__':
    # Liczba cech w zbiorze danych.
    FEATURE_COUNT = 2

    # Liczba węzłów w warstwie ukrytej.
    HIDDEN_NODE_COUNT = 2
    # Liczba iteracji nauki sieci neuronowej.
    EPOCHS = 100


    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [0,0],

    ])


    Y = np.array([
        [0],
        [1],
        [1],
        [0]])

    # Uruchamianie sieci neuronowej.
    loss_history,classification_error_history,weights1_history,weights2_history =  run_neural_network(X,
                       Y,
                       FEATURE_COUNT,
                       HIDDEN_NODE_COUNT,
                       EPOCHS)
#    print(loss_history)
    # Wykres historii błędu MSE
    plt.figure(figsize=(18, 4))
    plt.subplot(1, 4, 1)
    plt.plot(loss_history)
    plt.title('MSE')
    plt.xlabel('Iteracja')
    plt.ylabel('MSE')

    # Wykres historii błędu klasyfikacji
    plt.subplot(1, 4, 2)
    plt.plot(classification_error_history)
    plt.title('błęd klasyfikacji')
    plt.xlabel('Iteracja')
    plt.ylabel('błęd klasyfikacji')

    # Wykresy historii wag pierwszej warstwy
    plt.subplot(1, 4, 3)
    weights1_history_np = np.array(weights1_history)  # Konwersja do NumPy dla łatwiejszego indeksowania
    for input_node in range(weights1_history_np.shape[2]):  # Iteracja przez każde wejście
        for hidden_node in range(weights1_history_np.shape[1]):  # Iteracja przez każdy neuron w warstwie ukrytej
            plt.plot(weights1_history_np[:, hidden_node, input_node], label=f'Waga {hidden_node + 1},{input_node + 1}')
    plt.title('Historia wag (Warstwa 1)')
    plt.xlabel('Iteration')
    plt.ylabel('Waga')
    plt.legend(loc='upper left')
    plt.legend()

    # Wykresy historii wag drugiej warstwy
    plt.subplot(1, 4, 4)
    weights2_history_np = np.array(weights2_history)  # Konwersja do NumPy dla łatwiejszego indeksowania
    for i in range(weights2_history_np.shape[1]):  # Zmiana tutaj na shape[1], aby iterować przez neurony
        plt.plot(weights2_history_np[:, i],
                 label=f'Waga {i + 1}')  # Zmiana tutaj, brak potrzeby indeksowania drugiego wymiaru
    plt.title('Historia wag (Warstwa 2)')
    plt.xlabel('Iteracja')
    plt.ylabel('Waga')
    plt.legend()

    plt.tight_layout()
    plt.show()