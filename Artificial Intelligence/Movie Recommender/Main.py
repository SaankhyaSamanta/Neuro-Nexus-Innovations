import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

#Medium
data = pd.read_csv('imdb_top_1000.csv')
X = np.array(data.Overview)

data = data[['Genre','Overview','Series_Title', 'Released_Year']]
#print(data.head())

def Transform(X):
    #BERT
    text_data = X
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(text_data, show_progress_bar=True)
    X = np.array(embeddings)
    return X


"""
#PCA Optional
X = np.array(embed_data)
n_comp = 5
pca = PCA(n_components=n_comp)
pca.fit(X)
pca_data = pd.DataFrame(pca.transform(X))
pca_data.head()
sns.pairplot(pca_data)
"""

X = Transform(X)
cos_sim_data = pd.DataFrame(cosine_similarity(X))


def Recommendations(index, print_recommendation_plots= True, print_recommendations_genres =True):
  index_recomm = cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:11]
  movies_recomm =  data['Series_Title'].loc[index_recomm].values
  result = {'Movies':movies_recomm,'Index':index_recomm}

  print("The watched movie is : %s(%s)" %(data['Series_Title'].loc[index], data['Released_Year'].loc[index]))
  print("Genres: %s" %(data['Genre'].loc[index]))
  print("Plot:\n %s" %(data['Overview'].loc[index]))
  print("Top 10 recommended movies are:")
  k=0
  for movie in movies_recomm:
      print("#%d %s(%s)" %(k+1, movie, data['Released_Year'].loc[index_recomm[k]]))

      if print_recommendations_genres==True:
          genre = data['Genre'].loc[index_recomm[k]]
          print("Genres: %s" %(genre))

      if print_recommendation_plots==True:
          plot = data['Overview'].loc[index_recomm[k]]
          print("Plot:\n %s" %(plot))

      k += 1
  return result

def Plot(RecomMovies):
    to_plot_data = cos_sim_data.drop(index,axis=1)
    plt.plot(to_plot_data.loc[index],'.',color='red')
    x = RecomMovies['Index']
    y = cos_sim_data.loc[index][x].tolist()
    m = RecomMovies['Movies']
    plt.plot(x,y,'.',color='navy',label='Recommended Movies')
    plt.title('Movie Watched: '+data['Series_Title'].loc[index])
    plt.xlabel('Movie Index')
    k=0
    for x_i in x:
      plt.annotate('%s'%(m[k]),(x_i,y[k]),fontsize=10)
      k=k+1

    plt.ylabel('Cosine Similarity')
    plt.ylim(0,1)
    plt.show()
    return
    
#Recommendations(2)

Exit = 1
while(Exit!=0):
    Movie= input("Enter a movie's name: ")
    found = 0
    for index in range(0, len(data)):
        if data['Series_Title'].loc[index]==Movie:
            found = 1
            break
    if found==1:
        RecomMovies = Recommendations(index)
        Plot(RecomMovies)
    else:
        print("Sorry Movie not in list")
    Exit = int(input("Enter 0 to exit, any other number to continue "))

