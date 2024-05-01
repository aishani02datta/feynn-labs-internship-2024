import shutil
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

# src_pth = r"E:\INTERNSHIPS\Feynn Labs Internships\Project 2\car details v3.csv"
# dest_pth = os.path.join(os.getcwd(), 'car details v3.csv')
# shutil.copy(src_pth, dest_pth)

# fetching dataset - 1
df1 = pd.read_csv('car details v3.csv')
# print(df1.head())

# fetching dataset - 2
df2 = pd.read_csv('City-wise EV charging stations available in India.csv')
# print(df2.head())

# fetching dataset - 3
df3 = pd.read_csv('EV category-wise distribution sales to Consumers as per 2-8-23.csv')
# print(df3.head())

# fetching dataset - 4
df4 = pd.read_csv('State-wise current sales of EV vehicles in various segments.csv')
# print(df4.head())

df5 = pd.read_csv('EV vehicles based on brands.csv')
# print(df5.head())


# checking the shape (# of rows and columns) of the datasets
# print('DF1 Shape: ', df1.shape)
# print('DF2 Shape: ', df2.shape)
# print('DF3 Shape: ', df3.shape)
# print('DF4 Shape: ', df4.shape)
# print('DF5 Shape: ', df5.shape)

# checking the info (columns, datatypes, nulls) of the datasets
# print(' <<< DATASET 1 -----------------------------------------------------------')
# print(df1.info())
# print(' <<< DATASET 2 -----------------------------------------------------------')
# print(df2.info())
# print(' <<< DATASET 3 -----------------------------------------------------------')
# print(df3.info())
# print(' <<< DATASET 4 -----------------------------------------------------------')
# print(df4.info())
# print(' <<< DATASET 5-----------------------------------------------------------')
# print(df5.info())
# getting a statistical summary of the datasets
# df1 = df1.describe()
# df2 = df2.describe()
# df3 = df3.describe()
# df4 = df4.describe()
# df5 = df5.describe()
# print('<<< DATASET 1 >>>', df1, '<<< DATASET 2 >>>', df2, '<<< DATASET 3 >>>', df3, '<<< DATASET 4 >>>', df4, '<<< DATASET 5 >>>', df5)

# 2 wheelers data visualization from dataset 4
# plt.figure(figsize=(6, 6))
# sns.barplot(data=df4, y=df4['State Name'].sort_values(ascending=True), x='Two Wheeler', palette='viridis')
# plt.ylabel('State', fontsize=14, family='serif')
# plt.xlabel('Number of EV: 2 Wheelers', family='serif', fontsize=14, labelpad=10)
# plt.xticks(family='serif')
# plt.yticks(family='serif')
# plt.title(label='State-wise current sales of Electric Vehicles (2 Wheelers) in India', weight=200, family='serif', size=15, pad=12)
# plt.show()

# # 3 wheelers data visualization from dataset 4
# plt.figure(figsize=(6, 6))
# sns.barplot(data=df4, y=df4['State Name'].sort_values(ascending=True), x='Three Wheeler', palette='viridis')
# plt.ylabel('State', fontsize=14, family='serif')
# plt.xlabel('Number of EV: 3 Wheelers', family='serif', fontsize=14, labelpad=10)
# plt.xticks(family='serif')
# plt.yticks(family='serif')
# plt.title(label='State-wise current sales of Electric Vehicles (3 Wheelers) in India', weight=200, family='serif', size=15, pad=12)
# plt.show()

# 4 wheelers data visualization from dataset 4
# plt.figure(figsize=(6, 6))
# sns.barplot(data=df4, y=df4['State Name'].sort_values(ascending=True), x='Four Wheeler', palette='viridis')
# plt.ylabel('State', fontsize=14, family='serif')
# plt.xlabel('Number of EV: 4 Wheelers', family='serif', fontsize=14, labelpad=10)
# plt.xticks(family='serif')
# plt.yticks(family='serif')
# plt.title(label='State-wise current sales of Electric Vehicles (4 Wheelers) in India', weight=200, family='serif', size=15, pad=12)
# plt.show()


# charging stations availability visualization from dataset 2
# plt.figure(figsize=(6, 6))
# sns.barplot(data=df2, y=df2['City/Highway'].sort_values(ascending=True), x='Charging Stations', palette='viridis')
# plt.ylabel('State', fontsize=14, family='serif')
# plt.xlabel('Total number of Charging Stations', family='serif', fontsize=14, labelpad=10)
# plt.xticks(family='serif')
# plt.yticks(family='serif')
# plt.title(label='Number of Charging Stations available in India', weight=200, family='serif', size=15, pad=12)
# plt.show()

# brand-wise count of EV models
# sns.catplot(data=df5, x='Brand', kind='count', palette='viridis', height=6, aspect=2)
# sns.despine(right=False, top=False)
# plt.tick_params(axis='x', rotation=40)
# plt.xlabel('Brand',family='serif', size=12)
# plt.ylabel('Count', family='serif', size=12)
# plt.xticks(family='serif')
# plt.yticks(family='serif')
# plt.title('Number of EV Models Manufactured by a Brand', family='serif', size=15)
# plt.show()

# # analysis of different segments of EVs from dataset 5
# x = df5['Segment'].value_counts().plot.pie(radius=2, cmap='viridis', startangle=0, textprops=dict(family='serif'), pctdistance=.5)
# plt.pie(x=[1], radius=1.2, colors='white')
# plt.title(label='Electric Vehicles of Different Segments in India', family='serif', size=15, pad=100)
# plt.ylabel('')
# plt.show()

# different body types EVs visualization from dataset 5
# x = df5['BodyStyle'].value_counts().plot.pie(radius=2, cmap='viridis', startangle=0, textprops=dict(family='serif'))
# plt.pie(x=[1], radius=1.2, colors='white')
# plt.title(label='Electric Vehicles of Different Body Types based on different EV Models in India', family='serif', size=15, pad=100)
# plt.ylabel('')
# plt.show()

# Correlation between selling price and other variables and plotting of correlation heatmap from dataset - 1
# numeric_columns = df1.select_dtypes(include=['int', 'float']).columns
# corr_matrix = df1[numeric_columns].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()


# Price trend analysis for specific models over time from dataset - 5
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df, x='year', y='selling_price', hue='name', marker='o')
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.title('Price Trend Analysis for Specific Models Over Time')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# brand-wise analysis of the number of seats
# sns.countplot(data=df5, x='Brand', hue='Seats', palette='viridis')
# sns.despine(right=False, top=False)
# plt.tick_params(axis='x', rotation=40)
# plt.xlabel('Brand',family='serif', size=12)
# plt.ylabel('Number of Seats', family='serif', size=12)
# plt.xticks(rotation=40, family='serif')
# plt.yticks(family='serif')
# plt.title('Brand-wise Analysis of the Number of Seats', family='serif', size=15)
# plt.show()

# # speed visualization from dataset 5
# plt.figure(figsize=(6, 8))
# sns.barplot(data=df5, x='TopSpeed', hue='Brand', palette='viridis')
# plt.xticks(family='serif')
# plt.yticks(family='serif')
# plt.xlabel('Max Speed', family='serif', size=12)
# plt.ylabel('Brand', family='serif', size=12)
# plt.title(label='Brand-wise Speed Comparison of EVs in India', family='serif', size=15, pad=12)
# plt.show()

# acceleration visualization from dataset 5
# plt.figure(figsize=(6, 8))
# sns.barplot(data=df5, x='Accel', hue='Brand', ci=None,  palette='viridis')
# plt.xticks(family='serif')
# plt.yticks(family='serif')
# plt.xlabel('Acceleration', family='serif', size=12)
# plt.ylabel('Brand', family='serif', size=12)
# plt.title(label='Acceleration of EVs in India', family='serif', size=15, pad=12)
# plt.show()

# brand-wise analysis of the range parameter
# plt.figure(figsize=(8, 6))
# sns.barplot(data=df5, x='Brand', hue ='Range', palette='viridis', ci=None)
# sns.despine(right=False, top=False)
# plt.xticks(rotation=40, ha='right', family='serif', size=10)
# plt.yticks(family='serif', size=10)
# plt.xlabel('Brand',family='serif', size=12)
# plt.ylabel('Range per km', family='serif', size=12)
# plt.title('Brand-wise Analysis of the Range Parameter', family='serif', size=15)
# plt.tight_layout()
# plt.show()

# Handling missing values
df5.replace('-', np.nan, inplace=True)  # Replace '-' with NaN for missing values
df5.dropna(inplace=True)

#Encoding the categorical features
df5['PowerTrain'].replace(to_replace=['RWD', 'FWD', 'AWD'], value=[0, 1, 2], inplace=True)

# RapidCharge feature
df5['RapidCharge'].replace(to_replace=['No', 'Yes'], value=[0, 1], inplace=True)


# Selecting features for building the model
X = df5[
    ['Accel_Sec', 'TopSpeed_KmH', 'Efficiency_WhKm', 'FastCharge_KmH', 'Range_Km', 'RapidCharge', 'Seats', 'PowerTrain',
     'PriceEuro']]
y = df5['PriceEuro']  # Target variable

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

depths = []
mae_scores = []

# Iterate over different depths of decision tree
for depth in range(1, 11):
    # Initialize and train the decision tree regressor
    clf = DecisionTreeRegressor(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = clf.predict(X_test)

    # Calculate MAE and store in lists
    depths.append(depth)
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(depths, mae_scores, marker='o', linestyle='-')
plt.title('Decision Tree MAE vs. Depth', size=15, family='serif')
plt.xlabel('Depth of Decision Tree', family='serif')
plt.ylabel('Mean Absolute Error', family='serif')
plt.xticks(range(1, 11), family='serif')
plt.yticks(family='serif')
plt.grid()
plt.tick_params(axis='both', direction='inout', length=6, color='purple', grid_color='lightgray', grid_linestyle='--')
plt.show()