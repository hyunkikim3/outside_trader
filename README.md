# outside_trader
CAPP 30122 Group Project

References

http://excelsior-cjh.tistory.com/entry/5-Pandas를-이용한-Naver금융에서-주식데이터-가져오기

http://estenpark.tistory.com/353

http://dongsamb.com/web-scraping-using-python/

https://github.com/UC-MACSS/persp-model_W18


File name change:

df_1hour_Feb.json -> data/dataframe/dataframe_02.json

df_1hour_Mar_07.json -> data/dataframe/dataframe_03.json

balance_Feb27_Mar07_final.json -> data/balance_report/balance_report_02-27-12-50.json

model_df_weiwei_0307_final.json -> data/dataframe/combined_dataframe.json

model_df_final.json -> data/dataframe/combined_dataframe.json



Django

We have created a Django app to display the results of models we've implemented with graphs 
and tables. 

To go on to the website:
 * Please run Django with "source myvenv/bin/activate"
 * Run the app locally with "python3 manage.py runserver"
 * Open the server "http://127.0.0.1:8000/" on the website


To load data: 
 * To populate the data we've scraped from the web and results of models, we've written a file "load_data.py"
   - from the shell
   "python3 load_data.py"
   
   - with Django activated state (Please run Django with "source myvenv/bin/activate")
   "python3 manage.py makemigrations main"
   "python3 manage.py migrate"
 
 
 To check up on data in Django:
  * To write queries and explore models we've implemented
  
    - with Django activated state (Please run Django with "source myvenv/bin/activate")
    "python3 manage.py shell"
    
    i.e. from main.models import MODEL_NAME
    

 
