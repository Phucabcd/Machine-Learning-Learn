import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc file bằng pandas
df = pd.read_csv('train.csv')

# Chọn toàn bộ columns
# all_columns = df.columns
# Chọn các cột chữ "object" , số "number"
categorical_cols = df.select_dtypes(include=['object']).columns

# Kiểm tra xem missing values trong dữ liệu dataset hiện tại
missing_cat = df[categorical_cols].isnull().sum()
missing_cat = missing_cat[missing_cat > 0].sort_values(ascending=False)
missing_count = missing_cat.count()

# print("Columns missing: ", missing_cat)
# print("Total colums missing: ", missing_count)
print("Trước khi fillna:", missing_cat)

# fillna: "fill N/A" điền vào chỗ đang trống (NaN/null)
for col in categorical_cols:
    mode = df[col].mode()[0]
    df[col].fillna(mode, inplace=True)
    df[col] = df[col].fillna(mode)
    
missing_cat_after = df[categorical_cols].isnull().sum()
missing_cat_after = missing_cat_after[missing_cat_after > 0]
print("Sau khi fillna: \n", missing_cat_after)
    



