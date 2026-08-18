import pandas as pd
import matplotlib.pyplot as plt

read_csv = pd.read_csv('csv/Advertising.csv')
# print(read_csv)

#Get radio & sales columns => simple regression
# X = read_csv[['Radio']]
# y = read_csv['Sales']
X = read_csv.values[:, 2]
y = read_csv.values[:, 4]

# plt.scatter(X, y, marker='o')
# plt.show()

def predict(new_radio, weight, bias):
    return weight * new_radio + bias

def cost_function(X, y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight*X[i] + bias)) ** 2
    return sum_error / n
    
def update_weight(X, y, weight, bias, learning_rate):
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2 * X[i] * (y[i] - (weight*X[i] + bias))
        bias_temp += -2 * (y[i] - (weight*X[i] + bias))
    weight -= (weight_temp / n) * learning_rate
    bias -= (bias_temp / n) * learning_rate
    return weight, bias

def train(X, y, weight, bias, learning_rate, iter):
    cost_history = []
    for i in range(iter):
        weight, bias = update_weight(X, y, weight, bias, learning_rate)
        cost = cost_function(X, y, weight, bias)
        cost_history.append(float(cost))
    return weight, bias, cost_history

weight, bias, cost_history = train(X, y, 0.3, 0.2, 0.0001, 50)
print("Result:")
print("weight:",weight)
print("bias:",bias)
print("cost_history:",cost_history)
print("Regression:")
print("Predicted sales:", predict(19, weight, bias))  # weight * 19 + bias


#Đồ thị đường hồi quy (Regression Line) chồng lên dữ liệu thực tế
# plt.figure(figsize=(8, 6))
# plt.scatter(X, y, marker='o', color='blue', label='Dữ liệu thực tế')
# Vẽ đường hồi quy
# X_line = [min(X), max(X)]
# y_line = [predict(x, weight, bias) for x in X_line]
# plt.plot(X_line, y_line, color='red', linewidth=2, label='Đường hồi quy')

# plt.xlabel('Radio Advertising')
# plt.ylabel('Sales')
# plt.title('Simple Linear Regression: Radio vs Sales')
# plt.legend()
# plt.show()


#Đồ thị Cost History (kiểm tra mô hình có hội tụ tốt không)
# plt.figure(figsize=(8, 6))
# plt.plot(cost_history)
# plt.xlabel('Iteration')
# plt.ylabel('Cost (MSE)')
# plt.title('Cost giảm dần qua các vòng lặp')
# plt.show()


# Vẽ cả 2 đồ thị cạnh nhau (subplot) so sánh
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Đồ thị 1: Regression line
# axes[0].scatter(X, y, marker='o', color='blue', label='Dữ liệu thực tế')
# X_line = [min(X), max(X)]
# y_line = [predict(x, weight, bias) for x in X_line]
# axes[0].plot(X_line, y_line, color='red', linewidth=2, label='Đường hồi quy')
# axes[0].set_xlabel('Radio')
# axes[0].set_ylabel('Sales')
# axes[0].set_title('Regression Result')
# axes[0].legend()

# # Đồ thị 2: Cost history
# axes[1].plot(cost_history, color='green')
# axes[1].set_xlabel('Iteration')
# axes[1].set_ylabel('Cost (MSE)')
# axes[1].set_title('Cost Convergence')

# plt.tight_layout()
# plt.show()