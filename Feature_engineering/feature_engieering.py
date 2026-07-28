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
    df[col] = df[col].fillna(mode)
    
missing_cat_after = df[col].isnull().sum()
missing_cat_after = missing_cat_after[missing_cat_after > 0]
print("Sau khi fillna: \n", missing_cat_after)

features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt', 'MSSubClass']  

# Plot box plots
df[features] = np.log(df[features])
df[features].boxplot(figsize=(8, 4))

plt.title('Box Plot for Outlier Detection')
plt.ylabel('Values')
plt.xticks(rotation=45)
# plt.show()

numerical_columns = df.select_dtypes(include=['number']).columns

# Kiểm tra trước
missing_num = df[numerical_columns].isnull().sum()
missing_num = missing_num[missing_num > 0].sort_values(ascending=True)
print("Trước khi fillna (numeric):\n", missing_num)

for col in numerical_columns:
    median = df[col].median()  
    df[col] = df[col].fillna(median)  # Replace nulls with median

# Kiểm tra sau
missing_num_after = df[numerical_columns].isnull().sum()
missing_num_after = missing_num_after[missing_num_after > 0]
print("Sau khi fillna (numeric):\n", missing_num_after)


# Get columns that contain 'Yr' or 'Year'
year_columns = [feature for feature in numerical_columns if 'Yr' in feature or 'Year' in feature]

# Convert year values into age-related features
for col in year_columns:
    df[col] = df['YrSold'] - df[col]

print(year_columns)

# có thể dùng lại ở trên, nhưng select lại sẽ rõ ràng và dữ liệu an toàn, ổn định hơn
numerical_columns = df.select_dtypes(include=['number']).columns

#df.loc[điều_kiện_chọn_DÒNG, điều_kiện_chọn_CỘT] || : là lấy tất cả 
# any() kểm tra xem trong 1 tập giá trị, có ít nhất 1 giá trị nào là True | example: 98 false 2 true = true
numerical_0s = df.loc[:, (df == 0).any()].select_dtypes(include=['number']).columns
numerical_columns = numerical_columns.difference(numerical_0s)

#skewness đo độ lệch | khá trựu tượng khó hiểu 
skewness = df[numerical_columns].skew()
skewed_columns = skewness[abs(skewness) < 1]
#Kiểm tra skewness từng cột sau đó log những cột > 1 
# |skew| < 0.5 → gần như đối xứng, không cần xử lý
# |skew| < 1 → lệch nhẹ, có thể cân nhắc xử lý hoặc không
# |skew| > 1 → lệch đáng kể, nên xử lý (ví dụ log)

print("Skewed Columns:")
print(skewed_columns)

skew_features = ['1stFlrSF', 'GrLivArea', 'LotArea', 'SalePrice']

for col in skew_features:
    df[col] = np.log(df[col])

print(df[skew_features].skew())


categorical_columns = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_columns:
    labels_ordered = df.groupby([col])['SalePrice'].mean().sort_values().index
    labels_ordered = {x: i for i, x in enumerate(labels_ordered, 0)}
    df[col] = df[col].map(labels_ordered)

# Chọn ngẫu nhiên 10 dòng để làm missing
random_indices = df.sample(10).index
df.loc[random_indices, 'SalePrice'] = np.nan
df.loc[df['YearBuilt'] < 50, 'GarageType'] = np.nan

print("Total Missing: ",df.isnull().sum().sum())  
print(df['SalePrice'].isnull().sum())  
print(df.head())