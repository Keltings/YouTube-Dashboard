#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import datetime


# In[4]:


#define functions


# In[9]:


# load data
@st.cache # a way to reloadthe stramlit once
def load_data():
    
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv').iloc[1:,:] #skipp the first row
    #df_agg.head()
    df_agg_sub = pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('All_Comments_Final.csv')
    df_time = pd.read_csv('Video_Performance_Over_Time.csv')
    
    return df_agg, df_agg_sub, df_comments, df_time


# In[10]:


df_agg, df_agg_sub, df_comments, df_time = load_data()


# In[ ]:




