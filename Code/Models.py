#Load in cleaned datasets
from Clean import books
from Clean import ratings
from Clean import users
from Clean import master
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###Demographic filtering (generalised, what is most popular)
#Getting total ratings
top5 = pd.DataFrame(master.groupby(['ISBN'])['Book-Rating'].sum())
#Getting number of times book has been rated
rating_count = master['ISBN'].value_counts()
rating_count = pd.DataFrame(rating_count).rename(columns={"ISBN": "Rating-Count"})
rating_count['ISBN'] = rating_count.index
#Merging with rating counts and books dataset so we can get counts and images of book covers for slides
top5 = pd.merge(top5, rating_count, on = 'ISBN')
top5 = pd.merge(books, top5, on = 'ISBN')
top5 = top5.sort_values('Book-Rating', ascending = False)
print(top5.head(5)['Book-Title'])

###Content based recommendation system
#Drop index
popular_book = master.reset_index(drop = True)
#Initialise TF-IDF with terms up to 4 words long, appear in at least one book title, and ignore typical words
tf = TfidfVectorizer(ngram_range=(1, 4), min_df = 1, stop_words='english')
#Create TF-IDF matrix where each row is a book and each column is a term
tfidf_matrix = tf.fit_transform(popular_book['Book-Title'])
#Convert to floats
normalized_df = tfidf_matrix.astype(np.float32)
#Create new dataframe that calculates cosine similarity between every book (row)
cosine_similarities = cosine_similarity(normalized_df, normalized_df)
#Input lord of the rings book
bookName = 'The Fellowship of the Ring (The Lord of the Rings, Part 1)'
#Number of books to output
number = 10

print("Recommended Books:\n")
#Get ISBN of lord of the rings book
isbn = books.loc[books['Book-Title'] == bookName].reset_index(drop = True).iloc[0]['ISBN']
content = []
#Get index of lord of the rings
idx = popular_book.index[popular_book['ISBN'] == isbn].tolist()[0]
#Get array of indices most similar to lord of the rings
similar_indices = cosine_similarities[idx].argsort()[::-1]
similar_items = []
for i in similar_indices:
    #Ensure not same lord of the rings book and not already in similar items and similar items not larger than total recommendations
    if popular_book['Book-Title'][i] != bookName and popular_book['Book-Title'][i] not in similar_items and len(similar_items) < number:
        similar_items.append(popular_book['Book-Title'][i])
        content.append(popular_book['Book-Title'][i])
#Print out results
for book in similar_items:
    print(book)
#Get images for slides
master[master['Book-Title'] == similar_items[0]]

#Collaborative filtering (user-based) recommendation system
matrix = master.pivot(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating').fillna(0)
#Model itself with metric as cosine similarities and brute force algorithm
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
#fitting model
model_knn.fit(matrix)
#Find index of lord of the rings book
matrix.loc[matrix.index.str.contains("Lord of the Rings", case=False)]
#apply model
lotr = 'The Fellowship of the Ring (The Lord of the Rings, Part 1)'
#getting both distance and index of 5 books closest to lord of the rings
distances,indices = model_knn.kneighbors(matrix.loc[lotr,:].values.reshape(1,-1), n_neighbors = 6)
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(lotr))
    else:
        #print book name and distance
        print('{0}: {1}, with distance of {2}'.format(i, matrix.index[indices.flatten()[i]], distances.flatten()[i]))

