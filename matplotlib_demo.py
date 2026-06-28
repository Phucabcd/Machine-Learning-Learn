import matplotlib.pyplot as plt

# x = [3, 5]
# y = [2, 7]

# plt.plot(x, y, marker='o', color='red')
# plt.title("Matplotlib ve do thi")
# plt.grid(True)
# plt.show()

# do thi hinh ngoi sao
x = [3, 5, 7, 3, 7, 3]
y = [1, 7, 1, 5, 5, 1]
# c = [2, 4, 6, 8, 9, 10, 12]

z = [4, 3, 6]
plt.plot(x, y, c)
plt.plot(z, color='green')
plt.show()

#plot 1 mang plot(mang[n]) so luong phan tu trong mang => hang(x), mang[n] => cot(y)
#plot 2 mang plot.(mang[value1], mang[value2]) value1 => hang(x), value2 => cot(y)

# x_sao = [3, 5, 7, 3, 7, 3]
# y_sao = [1, 7, 1, 5, 5, 1]

# x_duong = [0, 4, 8]
# y_duong = [2, 5, 8]

#Truyen nhieu mang 
# plt.plot(x_sao, y_sao, x_duong, y_duong)
# plt.show()