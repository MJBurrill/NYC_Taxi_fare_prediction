# NYC_Taxi_fare_prediction
Final code used for the Kaggle Data science Competition  
link to contest: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction

Placed 81/1488 teams

Data provided:
  - pickup_datetime 
  - pickup_longitude 
  - pickup_latitude 
  - dropoff_longitude 
  - dropoff_latitude 
  - passenger_count 
  - fare
  
  From this, the competition was to predict the fare given the features above without the fare of course.
  
#Feature Engineering Thoughts:

-Intuitively thought that passenger count was probably negligible. I found this be true by both looking at the relationship between passenger count and fare as well as RMSE score with and without the passenger count variable when run through the model. As a result, drop this from the beginning to increase computation speed and the number of data points that can be run through the model using kaggle’s online computation tool.

-Break down pickup_datetime into as many components as possible. Broken down into hour, day, weekday, month, year.

-Distance between points. Easiest way is just taking the absolute value of the difference between pick and drop off locations for both latitude and longitude.  Also learned about Haversine distance from this kernel, which proved very helpful: https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration . I ended up using the function from this kernel however https://www.kaggle.com/gunbl4d3/xgboost-ing-taxi-fares . Both haversine and the absolute value of the difference were used.

- Many kernels suggested that airports drop-offs and pickups were important to distinguish because the rides had additional costs.  I Initially wrote a function that spit out a Boolean value if the pick up or drop off was within a certain radius of the airport. I then found that this kernel https://www.kaggle.com/gunbl4d3/xgboost-ing-taxi-fares which instead gave the distance of the pick up or drop off to a certain point and found this superior. 
Not only was this great for the airports, but I realized that I could also use this to flag high traffic areas as we all know that fare increases with time and traffic increases time. 

 
#Model Thoughts:

-As I didn’t want to spend any money on computing time I limited myself to only using Kaggle's free cloud computing. This is limiting because you have memory and time restrictions.

-I started by using a random forest as I had used this in another project before and had marginal results. After some research and looking through kernels I started using XGBoost, this kernel was very helpful. https://www.kaggle.com/gunbl4d3/xgboost-ing-taxi-fares. I saw some good improvement, but was limited to only running about 4 million records of the 55 million available. Finally, I stumbled upon this kernel which introduced me to LGBM https://www.kaggle.com/jsylas/python-version-of-top-ten-rank-r-22-m-2-88. This allowed me to run 22-25 million records depending on how many variables I had.
