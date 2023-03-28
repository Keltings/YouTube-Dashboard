# import libraries
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import datetime
from datetime import datetime


# define functions to format the datafame
def style_negative(v, props=''):
     """style negative values in df"""
     try:
          return props if v < 0 else None
     except:
          pass  

def style_positive(v, props=''):
     """style positive values in df"""
     try:
          return props if v > 0 else None
     except:
          pass

# function splits into Us India or other
def audience_simple(country):
     """Show top countries"""
     if country == 'US':
          return 'USA'
     elif country == 'IN':
          return 'India'
     else:
          return 'Other'
  

# load data
@st.cache_data # a way to reloadthe streamlit once
def load_data():
    
    df_agg = pd.read_csv('Aggregated_Metrics_By_Video.csv').iloc[1:,:] #skipp the first row
    df_agg.columns = ['Video','Video title','Video publish time','Comments added','Shares','Dislikes','Likes',
                      'Subscribers lost','Subscribers gained','RPM(USD)','CPM(USD)','Average % viewed','Average view duration',
                      'Views','Watch time (hours)','Subscribers','Your estimated revenue (USD)','Impressions','Impressions ctr(%)']
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'])
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    df_agg['Engagement_ratio'] =  (df_agg['Comments added'] + df_agg['Shares'] +df_agg['Dislikes'] + df_agg['Likes']) /df_agg.Views
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg.sort_values('Video publish time', ascending = False, inplace = True)    
    df_agg_sub = pd.read_csv('Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')
    df_comments = pd.read_csv('Aggregated_Metrics_By_Video.csv')
    df_time = pd.read_csv('Video_Performance_Over_Time.csv')
    df_time['Date'] = pd.to_datetime(df_time['Date'])
    return df_agg, df_agg_sub, df_comments, df_time 

#create dataframes from the function 
df_agg, df_agg_sub, df_comments, df_time = load_data()

############################## engineer data########################

# get the aggregated differential for all of the data
df_agg_diff = df_agg.copy() #doesnt mess up the original df_agg
#make sure the data is from the most recent 12 months
metric_data_12mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12)
#print(df_agg_diff.head())
#get the median aggregate
median_agg = df_agg_diff[df_agg_diff['Video publish time'] >= metric_data_12mo].median()
#the above line can be done alternatively:
numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
# feed the list into our dataframe
df_agg_diff.iloc[:, numeric_cols] = (df_agg_diff.iloc[:,numeric_cols] - median_agg).div(median_agg)

# merge the time data merge dailly with publish datas to get delta
df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video', 'Video publish time']], left_on = 'External Video ID', right_on='Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

#get last 12 mo of data rather than all data
date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months=12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

#get daily view data(first 30), median & percentiles the below code could be done by groupby 
# but we use pivot tables
views_days = pd.pivot_table(df_time_diff_yr, index='days_published', values='Views', aggfunc=[np.mean,np.median,lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published','mean_views','median_views','80pct_views','20pct_views']
#take the days between 0 and 30
views_days = views_days[views_days['days_published'].between(0,30)]
views_cumulative = views_days.loc[:,['days_published','median_views','80pct_views','20pct_views']]
views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()

## build dashboard
add_sidebar = st.sidebar.selectbox('Aggreggate or Individual Video', ('Aggregate Metrics', 'Individual Video Analysis'))

## Total picture
if add_sidebar == 'Aggregate Metrics':
     #bit of data enginering
     # metrics that are relevant in descending order
     df_agg_metrics = df_agg[['Video publish time','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]
     # figure the most 06/12 month date range 
     metric_data_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6)
     metric_data_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=12)
     #take the aggregate of all of the medians
     metric_median6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_data_6mo].median()
     metric_median12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_data_12mo].median()
     
     # use the individual metric counts
     #st.metric('Views', metric_median6mo['Views'], 500) # the 500 creates an arrow that increases by the metric

     # instead of metrics we could use columns
     # create 5 columnsin the stramlit dashboard
     col1, col2, col3, col4, col5 = st.columns(5)
     columns = [col1, col2, col3, col4, col5]

     # loop through all the diff columns
     count = 0
     for i in metric_median6mo.index:
          with columns[count]:
               delta = (metric_median6mo[i] - metric_median12mo[i])/metric_median12mo[i]
               st.metric(label= i, value = round(metric_median6mo[i],1), delta = '{:.2%}'.format(delta))
               count += 1
               if count >= 5:
                    count = 0

     # add df to the dashboard
     #get date information / trim to relevant data 
     df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())
     df_agg_diff_final = df_agg_diff.loc[:,['Video title','Publish_date','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]              
     
     # format the columns individually as %age
     #get a list of the numeric columns
     df_agg_numeric_lst = df_agg_diff_final.median().index.tolist()
     # for each of this columns has the format belo
     df_to_pct = {}
     for i in df_agg_numeric_lst:
          df_to_pct[i] = '{:.1%}'.format 

     st.dataframe(df_agg_diff_final.style.applymap(style_negative, props='color:red;').applymap(style_positive, props='color:green;').format(df_to_pct))


#individual dropdown
if add_sidebar == 'Individual Video Analysis':
     #getthe videos as tuple
     videos = tuple(df_agg['Video title'])
     # make the drop down
     video_select = st.selectbox('Pick A Video', videos)

     ##create the barchart 
     # by filtering our data based on the video picked
     agg_filtered = df_agg[df_agg['Video title'] == video_select]
     agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]
     agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)
     agg_sub_filtered.sort_values('Is Subscribed', inplace=True)

     #build 1st graph using plotly express
     fig = px.bar(agg_sub_filtered, x = 'Views', y = 'Is Subscribed', color='Country', orientation='h')

     #display figure
     st.plotly_chart(fig)

     #build 2nd  chart using line
     agg_time_filtered = df_time_diff[df_time_diff['Video Title']==video_select]
     first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
     first_30 = first_30.sort_values('days_published')

     fig2 = go.Figure()
     #add each individual line into this rather than allowing px to format for us
     fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                    mode='lines',
                    name='20th percentile', line=dict(color='purple', dash ='dash')))
     fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                        mode='lines',
                        name='50th percentile', line=dict(color='black', dash ='dash')))
     fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                        mode='lines', 
                        name='80th percentile', line=dict(color='royalblue', dash ='dash')))
     fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                        mode='lines', 
                        name='Current Video' ,line=dict(color='firebrick',width=8)))
        
     fig2.update_layout(title='View comparison first 30 days',
                   xaxis_title='Days Since Published',
                   yaxis_title='Cumulative views')
    
     st.plotly_chart(fig2)

