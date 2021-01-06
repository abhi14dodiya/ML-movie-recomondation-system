import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer#will allow by scikit learn to count sawords in sentance or array
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movie_dataset.csv")
#print(df.head())#will print few upper data#this is just for obervation
#print(df.shape)#will print size #this is just for obervation

cv = CountVectorizer()
features = ["keywords","cast","genres","director"]

for feature in features:
    df[feature] = df[feature].fillna('')# doing this because NAN value will effect in executon so we replace with null

def combine_feature(row):
    try:
        return row["keywords"] +" "+ row["cast"] + " "+ row["genres"] + " "+ row["director"]
    except:
        print(row)
df["combined_features"] = df.apply(combine_feature,axis=1)        
#this is just for obervationprint(df.head())#this is just for obervation
z = df.iloc[0]["combined_features"]#will take input as row if we encounter multipal input
#print(z)#features of that [0] movie
#find cosine similarities

count_matrix = cv.fit_transform(df["combined_features"])
cosine_model = cosine_similarity(count_matrix)
cosine_model_df = pd.DataFrame(cosine_model,index=df.title,columns=df.title)#this will add additional index in x and y axis as title
print(cosine_model_df.head())#wil print each movie is how much similar to other movie.

def make_recommondation(movie_user_like):
    return cosine_model_df[movie_user_like].sort_values(ascending=False)[0:10]#will sort certain column in desending so we can get most related movies

mv = input("enter related watched movie name: ")
z = make_recommondation(mv)
print(z)#will send request to function
