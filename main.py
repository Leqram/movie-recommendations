import pandas as pd
import matplotlib.pyplot as plt
import ssl

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage import io

ssl._create_default_https_context = ssl._create_unverified_context




movies_df = pd.read_csv('my.csv')
movies_df.drop_duplicates(inplace=True, ignore_index=True)


movies_df.fillna(value={i: ' ' for i in ['overfiew', 'genres', 'keywords', 'credits']}, inplace=True)

strOp = lambda x: ' '.join(x.split('-'))
# def strOp(x):
#     return ' '.join(x.split('-'))

movies_df.overview = movies_df.overview + movies_df.keywords.apply(strOp) + movies_df.genres.apply(strOp)
+ movies_df.credits.apply(lambda x: ' '.join(x.replace(' ', '').split('-')[:3]))

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies_df['overview'].values.astype('U'))

# print(pd.DataFrame(
#     tfidf_matrix[:10, 7000:7070].toarray(),
#     columns= tfidf.get_feature_names()[7000:7070],
#     index = movies_df.title[:10]).round())


def get_recomendations(title):
    idx = movies_df.index[movies_df['title'] == title][0]
    try:
        a = io.imread(f'https://image.tmdb.org/t/p/w500/{movies_df.loc[idx, "poster_path"]}', plugin='pil')
        plt.imshow(a)
        plt.axis('off')
        plt.title(title)
        plt.show()
    except Exception as e:
        print(f"error: {e}")
    
    
    print('Recomendations\n')

    sim_scores = list(enumerate(
        cosine_similarity(
            tfidf_matrix,
            tfidf_matrix[idx]
        )
    ))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:10]
    movie_indicates = [i[0] for i in sim_scores]

    result = movies_df.iloc[movie_indicates]

    fig, ax = plt.subplots(3, 3, figsize=(15, 20))
    ax = ax.flatten()
    for i, j in enumerate(result.poster_path):
        try:
            ax[i].axis('off')
            ax[i].set_title(result.iloc[i].title, fontsize=22)
            a = io.imread(f"https://image.tmdb.org/t/p/w500/{j}", plugin='pil')
            ax[i].imshow(a)
        except Exception as e:
            print(e)
    fig.tight_layout()
    # fig.show()
    plt.show()

get_recomendations("Batman")
