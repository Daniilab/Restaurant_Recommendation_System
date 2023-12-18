# Restaurant Recommendation System using Collaborative and Content filtering


# Description: 

This machine learning project leverages the Yelp Open Dataset and recommends users new restaurants in the Santa Barbara area based on their learned tastes and requested atmosphere. This recommendation system is a hybrid model that combines collaborative and content filtering. 

Link to dataset: https://www.yelp.com/dataset

#### Collaborative filtering:

This model uses matrix factorization with stochastic gradient descent to predict what users would rate every restaurant in the dataset. To generate predictions for a new user, the user must input a list of known liked restaurants. The model infers the user’s liked latent features based on their liked restaurants and is able to generate a predicted star rating (1-5 stars) for every restaurant in the dataset.  

#### Content filtering:

The user is able to filter for restaurants based on ambiance/occasion. This was achieved through creating a dictionary of keywords found in users' reviews. For each keyword, there is a list of restaurants associated with that keyword in the dictionary. 

#### Hybrid Model Reccomendation System:

I combined the collaborative and content filtering as follows: The collaborative filtering model makes predictions for what a user would rate every restaurant in the dataset. Of those restaurants, restaurants that do not contain the specified keyword are filtered out. The remaining restaurants are then sorted in descending value by predicted rating, and the top 10 restaurants are recommended.  


# Usage Instructions: 

## If you just want to generate recommendations:
The only file needed to generate recommendations is 6_hybrid_reccs.ipynb. Input a set of known liked restaurants in the final cell of 6_hybrid_reccs.ipynb. Then, run the entire script, and you will be prompted to enter the desired atmosphere out of a set of options. Once you enter the desired keyword, a list of 10 personalized recommendations will be generated.

## If you want to see the entire pipeline starting from raw data to recommendations: 

Start with script 2, then move to 3, 4, 5, 6. These scripts display the order of my work as I moved through the data pipeline (script 1 was exploratory and was not used in the pipeline).


** Note: There are 3 data files that I used that were too big to upload to Git Hub. The first file is the Yelp Dataset (link provided at the top). The other two are prepped_data_n_5_r.csv (created in second script) and business_reviews_nlp (created in 5th script. This shouldn't be a problem since they are not used in a script other than the one they were created in (they were only created to save time of running the program). **

# File Organization 

The ipynb files are organized in the order of the progress of my project. I started with exploratory data analysis, then moved to data preprocessing, then built my matrix factorization model and tuned hyperparameters, then created the collaborative filtering recommender system, and finished by creating a hybrid model that incorporates content-based filtering. 

- 1_yelp_EDA.ipynb displays my initial exploratory data analysis
- 2_user-item-matrix.ipynb displays my data preprocessing of cleaning, filtering, and wrangling data into a user-item matrix
- 3_model_building.ipynb is where I trained my collaborative filtering matrix factorization model and tuned the hyper parameters
- 4_reccomendation_system.ipynb is where I created my collaborative filtering MVP recommender system. In this script you can enter known liked restaurants and generate recommendations. I also added a rough draft version of the hybrid system here.
- 5_theme_analysis.ipynb contributes to the content filtering part of the model. This script is where I created a keyword_to_restaurant_ids dictionary and wrote to a JSON file to be accessed in the hybrid model 
- 6_hybrid_reccs.ipynb is my final product. This incorporates all the work from scripts 1 - 5.


# Notes

Initially I had planned to utilize NLP within the content based filtering. My plan was to filter for reviews for each restaurant related to ambiance, summarize them into a paragraph using a hugging face summarization model, and use Open AI’s embeddings to vectorize a summary for each restaurant. Following that, I planned to have a user input a sentence that described a desired ambiance, vectorize the description, and recommend restaurants that had summary descriptions with the closest cosine similarity to the desired ambiance. However, I discovered that this method was unlikely to produce effective results due to the difficulty of creating restaurant summaries–only a small fraction of review text is relevant to ambiance / occasion, and thus there is a lot of noise. Given the time constraints of the project, I decided upon using key-word filtering as it is a much simpler, time efficient, and effective method for capturing ambiance / occasion. 
