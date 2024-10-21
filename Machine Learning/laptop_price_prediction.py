import seaborn as sns 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df=pd.read_csv('/kaggle/input/real-world-laptop-data-analysis/laptop_uncleaned.csv')
df.head()

df.info()

df.isnull().sum()

df=df.rename(columns=str.lower)
df.drop(columns=['title','resolution','weight','usb','battery'],axis=1,inplace=True)
df.head()

mask=df['screen_size']=='1 Centimeters'
df_1_cm=df[mask]
df_1_cm

df=df[~mask]
df

df['cpu_model'].value_counts()

df['brand'].value_counts(normalize=True)*100

count_top10=df['brand'].value_counts().head()
fig,axes=plt.subplots(1,2,figsize=(12,8))
count_top10.plot.pie(autopct='%1.1f%%',ax=axes[0])
axes[0].set_title('Distribution of Brand(Top 10)')
count_top10.plot.barh(ax=axes[1])
axes[1].set_title('Brand Bar Plot')
axes[1].set_ylabel('Brand')
axes[1].set_xlabel('Counts')
axes[1].tick_params(axis='y',labelsize=8)
plt.tight_layout()
plt.show()

cpu_top10=df['cpu_model'].value_counts().head(10)
fig,axes=plt.subplots(1,2,figsize=(12,8))
cpu_top10.plot.pie(autopct='%1.1f%%',ax=axes[0])
axes[0].set_title('Distribution of CPU (Top 10)')
cpu_top10.plot.barh(ax=axes[1])
axes[1].set_title('CPU Bar Plot')
axes[1].set_ylabel('CPU')
axes[1].set_xlabel('Clounts')
axes[1].tick_params(axis='y',labelsize=8)
plt.tight_layout()
plt.show()

operating_system_top10=df['operating_system'].value_counts().head(10)
fig,axes=plt.subplots(1,2,figsize=(12,8))
operating_system_top10.plot.pie(autopct='%1.1f%%',ax=axes[0])
axes[0].set_title('Distribution of OPerating System (Top 10)')
operating_system_top10.plot.barh(ax=axes[1])
axes[1].set_title('Operating system top10 Bar Plot')
axes[1].set_ylabel('Operating System')
axes[1].set_xlabel('Counts')
axes[1].tick_params(axis='y',labelsize=8)
plt.tight_layout()
plt.show()

series_top10=df['series'].value_counts().head(10)
fig,axes=plt.subplots(1,2,figsize=(12,8))
series_top10.plot.pie(autopct='%1.1f%%',ax=axes[0])
axes[0].set_title('Distribution of Series (Top 10)')
series_top10.plot.barh(ax=axes[1])
axes[1].set_title('Series top10 Bar Plot')
axes[1].set_ylabel('Series')
axes[1].set_xlabel('Counts')
axes[1].tick_params(axis='y',labelsize=8)
plt.tight_layout()
plt.show()

mask_price = df['price'].isna()
df_with_nan = df[mask_price]
df_with_nan

# Convert 'Price' to numeric, handling errors
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['price'] = pd.to_numeric(df['price'], errors='coerce')


# Ensure 'Screen_Size', 'Ram', 'Disk_size', and 'Card_desc' are of type string
df['screen_size'] = df['screen_size'].astype(str)
df['ram'] = df['ram'].astype(str)
df['disk_size'] = df['disk_size'].astype(str)
df['card_desc'] = df['card_desc'].astype(str)

# Convert 'Screen_Size' to numeric
df['screen_size'] = df['screen_size'].str.replace(' Inches', '', regex=False).astype(float)

# Get top 10 values for each categorical variable
top_rams = df['ram'].value_counts().head(10)
top_disk_sizes = df['disk_size'].value_counts().head(10)
top_card_descs = df['card_desc'].value_counts().head(10)

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Plot distribution of Ratings
sns.histplot(df['rating'], ax=axes[0, 0], kde=True)
axes[0, 0].set_title('Distribution of Ratings')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frequency')

# Plot distribution of Screen Sizes
sns.histplot(df['screen_size'], ax=axes[0, 1], kde=True)
axes[0, 1].set_title('Distribution of Screen Sizes')
axes[0, 1].set_xlabel('Screen Size (Inches)')
axes[0, 1].set_ylabel('Frequency')

# Plot distribution of top RAM sizes
sns.barplot(x=top_rams.index, y=top_rams.values, ax=axes[1, 0])
axes[1, 0].set_title('Top 10 RAM Sizes')
axes[1, 0].set_xlabel('RAM Size')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot distribution of top Disk Sizes
sns.barplot(x=top_disk_sizes.index, y=top_disk_sizes.values, ax=axes[1, 1])
axes[1, 1].set_title('Top 10 Disk Sizes')
axes[1, 1].set_xlabel('Disk Size')
axes[1, 1].set_ylabel('Count')
axes[1, 1].tick_params(axis='x', rotation=45)

# Plot distribution of top Card Descriptions
sns.barplot(x=top_card_descs.index, y=top_card_descs.values, ax=axes[2, 0])
axes[2, 0].set_title('Top 10 Card Descriptions')
axes[2, 0].set_xlabel('Card Description')
axes[2, 0].set_ylabel('Count')
axes[2, 0].tick_params(axis='x', rotation=45)

# Plot distribution of Prices
sns.histplot(df['price'], ax=axes[2, 1], kde=True)
axes[2, 1].set_title('Distribution of Prices')
axes[2, 1].set_xlabel('Price')
axes[2, 1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

df.dropna(subset=['price'], inplace=True)
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


