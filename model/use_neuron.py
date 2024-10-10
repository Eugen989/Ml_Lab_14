import numpy as np
from neuron import SingleNeuron

# Загрузка весов из файла и тестирование
new_neuron = SingleNeuron(input_size=3)
new_neuron.load_weights('neuron_weights.txt')

# Пример использования
test_data = np.array([[9, 9, 9]])
predictions = new_neuron.feedforward(test_data[0])
# print(predictions, " ", type(predictions), " - ", predictions >= 0.5);
print("Предсказанные значения:", predictions, np.where(predictions >= 0.5, "Помидор", "Огурец"))