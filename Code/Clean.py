#Loading in packages
import numpy as np
import pandas as pd
ratings = pd.read_csv("Data/BX-Book-Ratings.csv")
books = pd.read_csv("Data/BX-Books.csv")
users = pd.read_csv("Data/BX-Users.csv")

print(ratings.shape)
print("hello world")