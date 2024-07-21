# Movie Recommendation System

This repository contains a simple content-based movie recommendation system using the TMDB 5000 Movies and Credits dataset. The system is built using Python and various libraries such as pandas, numpy, and scikit-learn.

## Table of Contents

1. [Dataset](#dataset)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Load the Data](#load-the-data)
    - [Select Relevant Columns](#select-relevant-columns)
    - [Data Preprocessing](#data-preprocessing)
    - [Convert JSON Columns](#convert-json-columns)
    - [Text Processing](#text-processing)
    - [Create Tags](#create-tags)
    - [Stemming](#stemming)
    - [Text Vectorization](#text-vectorization)
    - [Cosine Similarity](#cosine-similarity)
    - [Recommendation Function](#recommendation-function)
4. [Contributing](#contributing)
5. [License](#license)

## Dataset

The dataset used in this project can be found [here](https://www.kaggle.com/tmdb/tmdb-movie-metadata). It consists of two files:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## Installation

To run this project, you'll need to have Python installed. You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn nltk
```

## Usage

### Load the Data

Load the movie and credit datasets and merge them on the movie title.

```python
import numpy as np
import pandas as pd

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
```

### Select Relevant Columns

Keep only the columns that will help in content-based filtering.

```python
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
```

### Data Preprocessing

- Check for null values and remove them.
- Check for duplicate values.

```python
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
```

### Convert JSON Columns

Convert the `genres`, `keywords`, `cast`, and `crew` columns from JSON format to a list of strings.

```python
import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
```

### Text Processing

- Split the `overview` column into words.
- Remove spaces from the `genres`, `keywords`, `cast`, and `crew` columns.

```python
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
```

### Create Tags

Combine `overview`, `genres`, `keywords`, `cast`, and `crew` into a single `tags` column.

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
```

### Stemming

Apply stemming to the `tags` column to normalize the words.

```python
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
```

### Text Vectorization

Convert the `tags` column into vectors using `CountVectorizer`.

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
```

### Cosine Similarity

Compute the cosine similarity between the vectors.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
```

### Recommendation Function

Create a function to recommend movies based on a given movie title.

```python
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

recommend('Avatar')
```

## Contributing

If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
