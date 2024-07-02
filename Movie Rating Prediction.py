import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv(r'C:\Users\sumedh hajare\Downloads\IMDb Movies India.csv', encoding='latin1')

# Basic data exploration
print(df.head())
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())

# Clean data
df.dropna(inplace=True)
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)
df['Year'] = df['Year'].str.replace('[()]', '', regex=True).astype(int)
df['Votes'] = df['Votes'].str.replace(',', '').astype(int)

# Genre analysis
genres = df['Genre'].str.split(', ', expand=True).stack().value_counts()
print("Top genres:", genres.head())

# Visualizations
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['Year'].value_counts().sort_index().index, y=df['Year'].value_counts().sort_index().values)
plt.title("Movies Released per Year")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='Duration')
plt.title("Distribution of Movie Durations")
plt.show()

# Word cloud for genres
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(genres)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Genre Word Cloud')
plt.show()

# Rating analysis
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Rating", bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.show()

# Plot of Genres
plt.figure(figsize=(10, 8))
sns.barplot(x=genres.index[:10], y=genres.values[:10])
plt.title('Top 10 Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Top directors and actors
top_directors = df['Director'].value_counts().head(10)
top_actors = pd.concat([df['Actor 1'], df['Actor 2'], df['Actor 3']]).value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_directors.index, y=top_directors.values)
plt.title('Top 10 Directors')
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=top_actors.index, y=top_actors.values)
plt.title('Top 10 Actors')
plt.xticks(rotation=45, ha='right')
plt.show()


# Prepare data for modeling
df['Director_Code'] = df['Director'].astype('category').cat.codes
df['Genre_Code'] = df['Genre'].astype('category').cat.codes
df['Actor_Code'] = (df['Actor 1'] + df['Actor 2'] + df['Actor 3']).astype('category').cat.codes

# Split data for training and testing
X = df[['Year', 'Duration', 'Director_Code', 'Genre_Code', 'Actor_Code']]
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")