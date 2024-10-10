import numpy as np
from neuron import SingleNeuron


# Пример данных (X - входные данные, y - целевые значения)
data = np.array([
  [-2, -1],  # Алиса
  [25, 6],   # Боб
  [17, 4],   # Чарли
  [-15, -6], # Диана
])
all_y_trues = np.array([
  1, # Алиса
  0, # Боб
  0, # Чарли
  1, # Диана
])

# Инициализация и обучение нейрона
neuron = SingleNeuron(input_size=2)
neuron.train(data, all_y_trues, epochs=5000, learning_rate=0.1)

# Сохранение весов в файл
neuron.save_weights('neuron_weights.txt')