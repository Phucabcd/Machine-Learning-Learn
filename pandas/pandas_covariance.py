import pandas as pd 
import numpy as np

# df = pd.read_csv('csv\\List_demo_covariance .csv', index_col=0)

#seed dat mot diem bat dau de pseudo-random
#khong can phai la 42 ma co the la bat ky so nguyen nao "no chi la diem bat dau" 
#So 42 noi tieng tu tieu thuyet "The Hitchhiker's Guide to the Galaxy" author Douglas Adams
# np.random.seed(42)
# df = pd.DataFrame(
#     np.random.rand(10, 3), columns=['A', 'B', 'C']
# )

# EXAMPLE 1
# df = pd.DataFrame({
#     "Salary": [1000, 1590, 3400, 5000],
#     "Experience": [1, 2, 3, 5]
# })

# # Covariance
# matrix = df.cov() 
# # Correlation
# # matrix = df.corr()  
# print(matrix)
# # print(df.corr(method='spearman'))

# EXAMPLE 2
# Giả sử ta có dữ liệu của 3 học sinh
# df = pd.DataFrame({
#     'Gio_Hoc': [5, 15, 34],       # Tính bằng GIỜ
#     'Phut_Hoc': [300, 900, 2040], # Vẫn là dữ liệu đó nhưng tính bằng PHÚT (nhân 60)
#     'Diem_Thi': [5, 7, 9]        # Điểm số
# })

# # 1. Xem ma trận Hiệp phương sai (cov)
# print("--- Ma trận COV ---")
# print(df.cov())

# # 2. Xem ma trận Tương quan (corr)
# print("\n--- Ma trận CORR ---")
# print(df.corr())
# print("\n--- Ma trận CORR (Spearman) ---")
# print(df.corr(method='spearman'))

# EXAMPLE 3
# data = {
#     'Chi_Tieu': [100000, 150000, 1000000],  # Số cuối tăng vọt (Outlier)
#     'Hai_Long': [1, 2, 5]
# }
# df = pd.DataFrame(data)

# print("--- Tương quan Pearson (Mặc định) ---")
# print(df.corr(method='pearson')) # Kết quả sẽ rất thấp (~0.5) do số 10 triệu làm lệch

# print("\n--- Tương quan Spearman ---")
# print(df.corr(method='spearman')) # Kết quả sẽ ra 1.0 vì thứ hạng tăng đều
