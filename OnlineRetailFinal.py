# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns

import squarify
from sklearn import preprocessing
import sklearn
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from matplotlib import pyplot as plt
import plotly.express as px
import streamlit as st
import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import numpy
from sklearn.cluster import AgglomerativeClustering

# Drag/Drop file===========================================================================================
data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])

# 1. Data Understanding/ Acquire==========================================================================

st.title('Data Science')
st.write("## Customer Segmentation Online Retail")

menu = ['Overview','Preprocessing & EDA','RFM Analyze & Evaluation', 'Conclusion and Suggestion']

choice = st.sidebar.selectbox('Menu',menu)