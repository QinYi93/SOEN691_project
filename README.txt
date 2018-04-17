Readme
———————
1. User-Based Collaborative Filtering:
python3 ./code/userBased.py ./data/ratings.csv ./data/toBeRated.csv cosine jaccard pearson

After the above command finish executing, it will provide result1.csv as the output file which will have the predicted ratings. Also, rmse_user.txt will be an output which shows the rmse obtained in all the three types of similarity and also shows which one is the best.

2. Item-Based Collaborative Filtering:
python3 ./code/itemBased.py ./data/rating.csv ./data/toBeRated.csv cosine jaccard pearson

After the above command finish executing, it will provide result2.csv as the output file which will have the predicted ratings. Also, rmse_item.txt will be an output which shows the rmse obtained in all the three types of similarity and also shows which one is the best.


