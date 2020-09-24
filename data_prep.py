 
import numpy as np
from numpy import count_nonzero
import pandas as pd
pd.options.mode.chained_assignment = None 
import pymysql.cursors
import re
from sqlalchemy import create_engine
from sqlalchemy.sql import select
engine = create_engine("mysql+pymysql://root:bf_op12345@localhost/dbpets")
con = engine.connect()
 

class prep_pets:
	''' Class that fetches and prepares data from the dbpets database ''' 
 	
	def check_sparsity(X):
		sparsity = 1.0 - ( count_nonzero(X) / float(X.size) )
		print('Sparsity check:',sparsity) 

    #Function created by other researchers to get length of time an animal is in the shelter
	def get_days_length(val):
	    val = str(val)
	    days = re.findall('\d*',val)[0]
	    try:
	        days = int(days)
	        if days <= 7:
	            return "0-7 days"
	        elif days <= 21:
	            return "1-3 weeks"
	        elif days <= 42:
	            return "3-6 weeks"
	        elif days <= 84:
	            return "7-12 weeks"
	        elif days <= 168:
	            return "12 weeks - 6 months"
	        elif days <= 365:
	            return "6-12 months"
	        elif days <= 730:
	            return "1-2 years"
	        else:
	            return "2+ years"
	    except:
	        return np.nan
	    
	 
    #Feed in data, do some feature engineering, create one string for each record.
	def produce_predictor_matrix():

		#Connect to mysql database 'dbpets' and retrive the 'records' table which is data used by the other researchers. Then create a dataframe called 'records' from that table to work with below.
		records  = pd.read_sql("select * from dbpets.records", con) 

		#Split up DF and extract target. Use code from students to extract target
		records['Target'] = 0
		adopt_mask = records['Outcome_Type'] == 'Adoption'
		records.loc[adopt_mask, 'Target'] = 1

		#Feature engineering by other researchers.	 
		#Length of days in shelter 
		records['DateTime_intake'] = pd.to_datetime(records['DateTime_intake'])
		records['DateTime_outcome'] = pd.to_datetime(records['DateTime_outcome'])
		records['DateTime_length'] = records['DateTime_outcome'] - records['DateTime_intake']
		records['Days_length'] = records['DateTime_length'].apply(prep_pets.get_days_length)		 

	    #My engineered feature 
		#Get the day month year of animal intake. convert each to string and use each result as a token. 
		records['Month_intake'] = pd.to_datetime(records['DateTime_intake']) 
		records['day_intake']= (records['Month_intake'].dt.day).astype(str) 
		records['month_intake']= (records['Month_intake'].dt.month).astype(str) 
		records['year_intake']= (records['Month_intake'].dt.year).astype(str) 

        #This is the feature set I will work with 
		df_embeddings=records[[ 'Name_intake',  
	       'Intake_Type', 'IntakeCondition',
	       'Animal_Type_intake', 'Sex', 'Age', 'Breed_intake', 'Color_intake',
	       'Age_upon_Outcome',    
	       'Days_length','day_intake','month_intake','year_intake']] 

		df_embeddings.replace(np.nan,'missing',regex=True,inplace=True)


        #Create one string from each record
		df_embeddings['list'] = df_embeddings['Name_intake'].str.cat(df_embeddings['Intake_Type'],sep=" ")\
		                       .str.cat(df_embeddings['IntakeCondition'],sep=" ")\
		                       .str.cat(df_embeddings['Animal_Type_intake'],sep=" ")\
		                       .str.cat(df_embeddings['Sex'],sep=" ")\
		                       .str.cat(df_embeddings['Age'],sep=" ")\
		                       .str.cat(df_embeddings['Breed_intake'],sep=" ")\
		                       .str.cat(df_embeddings['Color_intake'],sep=" ")\
		                       .str.cat(df_embeddings['Age_upon_Outcome'],sep=" ")\
		                       .str.cat(df_embeddings['Days_length'],sep=" ")\
		                       .str.cat(df_embeddings['day_intake'],sep=" ")\
		                       .str.cat(df_embeddings['month_intake'],sep=" ")\
		                       .str.cat(df_embeddings['year_intake'],sep=" ") 


        #Save a record of the feature set used in my predictor matrix to the mysql dbpets data base. Resulting table is called 'df_embeddings'.
		df_embeddings['list'].to_sql(name='df_embeddings',con=con,if_exists='replace')

		#Return the X matrix as a list.  
		return df_embeddings['list'].astype(str),records['Target']






	 
