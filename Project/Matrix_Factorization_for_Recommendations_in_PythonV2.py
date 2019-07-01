import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

ratings_list = [i.strip().split("::") for i in open('ml-1m/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('ml-1m/movies.dat', 'r').readlines()]

ratings = np.array(ratings_list)
users = np.array(users_list)
movies = np.array(movies_list)

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int).astype(int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric).astype(int)

#print movies_df.head()
#print ratings_df.head()


R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0).astype(int)
df_p = pd.pivot_table(ratings_df,values='Rating',index='UserID',columns='MovieID')
#print R_df.head()

#This part of the code takes away the year of the movie from the string (For Pearson R)
newMovieColumn = []
for movie in movies_df["Title"]:
   newTitle = movie[0:len(movie)-7]
   newMovieColumn.append(newTitle)
movies_df["Title"] = newMovieColumn



def recommendMovieFromMovie(movie_title):
    row=movies_df.loc[movies_df["Title"]==movie_title]
    index = int(row["MovieID"])
    target = df_p[index]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target = corr_target.join(movies_df)[['PearsonR', 'Title']]
    print corr_target.head(50)


#user_ratings_mean = np.mean(R_df.values(), axis = 1)
#R_demeaned = R_df.values() - user_ratings_mean.reshape(-1, 1)

R = R_df.to_numpy()

user_ratings_mean = np.average(R.astype(int), axis = 1)
R_demeaned = R.astype(int) - user_ratings_mean.reshape(-1, 1)


#from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns).astype(int)
#print preds_df.head()


def recommend_movies(preds_df, userID, movies_df, ratings_df, num_recommendations=5):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)  # UserID starts at 1

    # Get the user's data and merge in the movie information.
    user_data = ratings_df[ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df.convert_objects(convert_numeric=True), how='left', left_on='MovieID', right_on='MovieID').
                 sort_values(['Rating'], ascending=False)
                 )

    print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
    print 'Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations)

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
                           merge(pd.DataFrame(sorted_user_predictions.convert_objects(convert_numeric=True)).reset_index(), how='left',
                                 left_on='MovieID',
                                 right_on='MovieID').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )

    return user_full, recommendations

#ratings for user 837
already_rated, predictions = recommend_movies(preds_df, 1310, movies_df, ratings_df, 10)

print already_rated.head(10)

print predictions

print("Movies recommendation for Jumanji")
recommendMovieFromMovie("Jumanji")
print("\n")
print("Movies recommendation for Runaway")
recommendMovieFromMovie("Runaway")