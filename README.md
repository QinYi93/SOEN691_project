## Readme
### project topic
1. Compare item-based and user-based CF
2. Test different similarity calculation method (Jaccard, Cosine, Person), and get a more accuracy result
3. predict to user for movies they didn't watch before 

### project contain
1. itembased_sp.py and userbased_sp.py calculate the RMSE value through spark on different density levels and make the recommendations.

2. itemBased.py and userBased.py calculate the RMSE value through python on different methods (jaccard, cosine, pearson).
* Run itemBased.py Please input follow in terminal  
python ./code/itemBased.py ./data/ratings.csv ./data/toBeRated.csv cosine jaccard pearson
* Run userBsed.py please input follow  
python ./code/userBased.py ./data/ratings.csv ./data/toBeRated.csv cosine jaccard pearson

3. result_analyst.py makes the recommendations based on the results which are calculated by pearson correlation.
