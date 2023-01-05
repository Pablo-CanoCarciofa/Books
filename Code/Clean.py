#Loading in packages
import numpy as np
import pandas as pd

#Loading in data
ratings = pd.read_csv("Data\BX-Book-Ratings.csv", sep = ";", on_bad_lines = 'skip', encoding = "latin-1")
books = pd.read_csv("Data\BX-Books.csv", sep = ";", on_bad_lines = 'skip', encoding = "latin-1", low_memory = False)
users = pd.read_csv("Data\BX-Users.csv", sep = ";", on_bad_lines = 'skip', encoding = "latin-1")

#Getting an idea of the data
ratings.head()
books.head()
users.head()

#Image URLs not useful for modelling so these are dropped from books dataset
books.drop(['Image-URL-S', 'Image-URL-M'], axis = 1, inplace = True)

### CLEANING RATINGS DATASET
#Checking object types of columns
ratings.dtypes
#Checking for outliers
len(ratings['User-ID'].unique())
ratings['Book-Rating'].unique()
#Checking for null values
ratings.loc[ratings['User-ID'].isnull()]
ratings.loc[ratings['Book-Rating'].isnull()]
#Need to ensure ratings are from users and books in respective datasets
ratings = ratings[ratings['ISBN'].isin(books['ISBN'])]
ratings = ratings[ratings['User-ID'].isin(users['User-ID'])]
#Only ratings above 0 (explicit) are included since they give most information
ratings = ratings[ratings['Book-Rating'] != 0] #this reduces dataset massively to around 380k ratings

### CLEANING BOOKS DATASET
#Checking object types of columns
books.dtypes
#Checking unique values of year of publication column
books['Year-Of-Publication'].unique()
#Found publishers in year of publication so checking these entries
books.loc[books['Year-Of-Publication'] == 'DK Publishing Inc']
books.loc[books['Year-Of-Publication'] == 'Gallimard']
#Ordering of data in columns is wrong, so fixing these entries (based on google search of book)
books.loc[books['ISBN'] == '078946697X', 'Book-Title'] = 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'
books.loc[books['ISBN'] == '078946697X', 'Book-Author'] = 'Michael Teitelbaum'
books.loc[books['ISBN'] == '078946697X', 'Year-Of-Publication'] = 2000
books.loc[books['ISBN'] == '078946697X', 'Publisher'] = 'DK Publishing Inc'
books.loc[books['ISBN'] == '0789466953', 'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'
books.loc[books['ISBN'] == '0789466953', 'Book-Author'] = 'James Buckley'
books.loc[books['ISBN'] == '0789466953', 'Year-Of-Publication'] = 2000
books.loc[books['ISBN'] == '0789466953', 'Publisher'] = 'DK Publishing Inc'
books.loc[books['ISBN'] == '2070426769', 'Book-Title'] = 'Peuple du ciel, suivi de \'Les Bergers'
books.loc[books['ISBN'] == '2070426769', 'Book-Author'] = 'Clézio Jean-Marie Gustave Le'
books.loc[books['ISBN'] == '2070426769', 'Year-Of-Publication'] = 2003
books.loc[books['ISBN'] == '2070426769', 'Publisher'] = 'Gallimard'
#Convert year of publication column to integer since some strings in there
books['Year-Of-Publication'] = np.floor(pd.to_numeric(books['Year-Of-Publication'], errors='coerce')).astype('Int64')
books.loc[(books['Year-Of-Publication'] > 2004) | (books['Year-Of-Publication'] == 0), 'Year-Of-Publication'] = np.NAN
books['Year-Of-Publication'].isna().sum() / books['Year-Of-Publication'].sum() #negligible number of nulls (4960)
#Check publisher column for empty values
books.loc[books['Publisher'].isnull()]
#Adding publishers to these rows
books.loc[books['ISBN'] == '193169656X', 'Publisher'] = 'Novelbooks Inc'
books.loc[books['ISBN'] == '1931696993', 'Publisher'] = 'Novelbooks Inc'

### CLEANING USERS DATASET
#Check age values
sorted(users['Age'].unique())
#Cut off ages below 2 and above 95
users.loc[(users['Age'] > 80) | (users['Age'] < 5),'Age'] = np.NAN
users['Age'] = users['Age'].astype('Int64')
#Check how many null values created
users['Age'].isna().sum() / users['Age'].sum() #approx 1/5 ages now null
#Not using location data to save memory
users.drop(['Location'], axis = 1, inplace = True)


#Combine book and rating datasets
master = pd.merge(ratings, books, on = 'ISBN')
master = pd.merge(users, master, on = 'User-ID')
#Reduce users to those who have rated over 100 books
user_count = master['User-ID'].value_counts()
master = master[master['User-ID'].isin(user_count[user_count >= 100].index)]
len(master['User-ID'].unique()) 
#Reduce books to those that have been rated over 10 times
rating_count = master['ISBN'].value_counts()
master = master[master['ISBN'].isin(rating_count[rating_count >= 10].index)]
len(master['ISBN'].unique())
#Create total rating column
rating_total = (master.groupby(by = ['Book-Title'])['Book-Rating'].sum().reset_index().rename(columns = {'Book-Rating':'Total-Rating'}))
master = master.merge(rating_total, left_on = 'Book-Title', right_on = 'Book-Title', how = 'left')
master = master.drop_duplicates(['User-ID', 'Book-Title'])
master.to_csv(r'master.csv')