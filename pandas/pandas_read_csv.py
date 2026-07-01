import pandas as pd
import numpy as np

df = pd.read_csv('habit_tracker_demo.csv')
# print(df.head(8))
#len => dem phan tu trong mang
print("Tổng việc cần làm:", len(df) * 5)

print("Công việc ngày thứ 3:", df.loc[3, "Completed"])
print("Số việc đã hoàn thành:", df["Completed"].sum())

# import 
# df.to_csv('result_habit_demo.csv')