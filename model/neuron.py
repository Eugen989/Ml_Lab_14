import numpy as np

def sigmoid(x):
  # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true и y_pred - массивы numpy одинаковой длины.
  return ((y_true - y_pred) ** 2).mean()


class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Умножаем входы на веса, прибавляем порог, затем используем функцию активации
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

# Нейрон
class SingleNeuron:
    def __init__(self, input_size):
        self.weights = np.array([0, 1, 2])
        self.bias = 0

        # Используем класс Neuron из предыдущего раздела
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()

        # Пороги
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        h3 = sigmoid(self.w5 * x[0] + self.w6 * x[1] + self.b3)
        o1 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.b4)
        return o1


    def forward(self, inputs):
        print(inputs, " - ", type(inputs))
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

    def backward(self, y, learning_rate=0.01):
        # Вычисляем ошибку
        error = self.output - y
        # Вычисляем градиенты
        d_output = error * sigmoid_derivative(self.output)
        d_weight = np.dot(self.input.T, d_output)
        d_bias = np.sum(d_output)

        # Обновление весов и смещения
        self.weight -= learning_rate * d_weight
        self.bias -= learning_rate * d_bias

    def train(self, data, all_y_trues, epochs=1000, learning_rate=0.01):
        learn_rate = 0.1
        epochs = 1000  # сколько раз пройти по всему набору данных

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Прямой проход (эти значения нам понадобятся позже)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_h3 = self.w5 * x[0] + self.w6 * x[1] + self.b3
                h3 = sigmoid(sum_h3)

                sum_o1 = self.w7 * h1 + self.w8 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Считаем частные производные.
                # --- Имена: d_L_d_w1 = "частная производная L по w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w7 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w8 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Нейрон h3
                d_h2_d_w5 = x[0] * deriv_sigmoid(sum_h3)
                d_h2_d_w6 = x[1] * deriv_sigmoid(sum_h3)
                d_h2_d_b3 = deriv_sigmoid(sum_h3)

                # --- Обновляем веса и пороги
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон h3
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b3

                # Нейрон o1
                self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_w7
                self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_w8
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Считаем полные потери в конце каждой эпохи
            # if epoch % 10 == 0:
            #     y_preds = np.apply_along_axis(self.feedforward, 1, data)
            #     loss = mse_loss(all_y_trues, y_preds)
            #     print("Epoch %d loss: %.3f" % (epoch, loss))

    def save_weights(self, filename):
        np.savetxt(filename, np.hstack((self.weight, self.bias)))

    def load_weights(self, filename):
        data = np.loadtxt(filename)
        self.weight = data[:-1]
        self.bias = data[-1]