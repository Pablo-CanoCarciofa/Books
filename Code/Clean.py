#Loading in packages
import numpy as np
import pandas as pd
ratings = pd.read_csv("Data\BX-Book-Ratings.csv", sep = ";", on_bad_lines = 'skip', encoding = "latin-1")
books = pd.read_csv("Data\BX-Books.csv", sep = ";", on_bad_lines = 'skip', encoding = "latin-1")
users = pd.read_csv("Data\BX-Users.csv", sep = ";", on_bad_lines = 'skip', encoding = "latin-1")
print(books.head())