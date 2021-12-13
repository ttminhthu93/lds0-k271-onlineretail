# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns

import squarify
from sklearn import preprocessing
import sklearn
# from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
import plotly.express as px
import streamlit as st
import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
# from collections import Counter
import numpy
from sklearn.cluster import AgglomerativeClustering
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

## Basic setup and app layout===============================================================================
# Header====================================================================================================
st.set_page_config(layout='wide')


# Sidebar ==================================================================================================
st.sidebar.title("Control Panel")


tick_size = 12
axis_title_size = 16

menu = ['Overview','Preprocessing & EDA','RFM Analyze & Evaluation', 'Conclusion and Suggestion']


## User inputs on the control panel
st.sidebar.subheader("Attributes Selection")
recency_sessions = st.sidebar.number_input(
    "Recency - click and select threshold",
    min_value=1,
    max_value=365,
    value=50,
    step=1,
    help="Recency - Lần mua hàng gần đây nhất của khách hàng.",
)

frequency_sessions = st.sidebar.number_input(
    "Frequency - click and select threshold",
    min_value=1,
    max_value=7491,
    value=100,
    step=1,
    help="Frequency - Tần xuất khách hàng mua hàng.",
)

monetary_sessions = st.sidebar.number_input(
    "Monetary - click and select threshold",
    min_value=1,
    max_value=250000,
    value=100,
    step=1,
    help="Monetary - Khách hàng chi bao nhiêu cho việc mua hàng.",
)
# Main Body ================================================================================================
st.title('Data Science')
st.write("## Customer Segmentation Online Retail")

st.write('- Attributes Selection:')
results = pd.DataFrame(
    {
        "Recency": [recency_sessions],
        "Frequency": [frequency_sessions],
        "Monetary": [monetary_sessions],
    },
    index=['Attributes'],
)
st.dataframe(results)

# posterior = copy.copy(prior)
# assert id(posterior) != id(prior)

st.write('- Classification Results:')
# Prepare data
df_kmean = pd.read_csv('Data/OnlineRetail_k4.csv', encoding= 'unicode_escape')
# Condition for attributes
def create_segment(df):
    if df['Recency'] >= 7 and df['Recency'] <= 365 | df['Frequency'] >= 1 and df['Frequency'] <= 74 | df['Monetary'] >= 3.75 and df['Monetary'] <= 4055.72:
        return 'LOST'
    elif df['Recency'] >= 1 and df['Recency'] <= 47 | df['Frequency'] >= 3 and df['Frequency'] <= 1491 | df['Monetary'] >= 414.30 and df['Monetary'] <= 231822.69:
        return 'NEW'
    elif df['Recency'] >= 1 and df['Recency'] <= 61 | df['Frequency'] >= 1 and df['Frequency'] <= 136 | df['Monetary'] >= 41.99 and df['Monetary'] <= 6748.80:
        return 'NEW'
    elif df['Recency'] >= 15 and df['Recency'] <= 353 | df['Frequency'] >= 1 and df['Frequency'] <= 521 | df['Monetary'] >= 202.27 and df['Monetary'] <= 77183.60:
        return 'RISK' 
results['Cluster'] = results.apply(create_segment, axis=1)
summary = df_kmean.groupby(['Cluster']).agg({'Recency': 'mean',
                                                    'Frequency': 'mean',
                                                    'Monetary': 'mean'}).round(0)
results_new = pd.merge(results ,summary , on=['Cluster','Cluster'])\
                .rename(columns={'Recency_y': 'Recency_Mean',
                                'Frequency_y': 'Frequency_Mean',
                                'Monetary_y': 'Monetary_Mean',
                                'Recency_x':'Recency',
                                'Frequency_x':'Frequency',
                                'Monetary_x': 'Monetary'
                                })                             
st.dataframe(results_new)
st.write('- Visualization:')


# # Drag/Drop file ===========================================================================================
# data = st.file_uploader("Upload a Dataset", type='csv')



  