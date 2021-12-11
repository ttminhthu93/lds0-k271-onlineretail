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
# from collections import Counter
import numpy
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

'test'
'test2'
# # Build project===========================================================================================
# data = pd.read_csv('Data/OnlineRetail.csv', encoding= 'unicode_escape')


# # 1. Data Understanding/ Acquire==========================================================================

# st.title('Data Science')
# st.write("## Customer Segmentation Online Retail")

# menu = ['Overview','Preprocessing & EDA','RFM Analyze & Evaluation', 'Conclusion and Suggestion']

# choice = st.sidebar.selectbox('Menu',menu)

# if choice == 'Overview':
#     st.subheader("I. Overview")
#     st.write("""
#     #### Business Objective/Problem:
#     - Công ty X chủ yếu bán các sản phẩm là quà tặng dành cho những dịp đặc biệt. Nhiều khách hàng của công ty là khách hàng bán buôn.
#     - Công ty X mong muốn có thể bán được nhiều sản phẩm hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.
#     """)
 
#     st.image("Data/audience-segmentation-m.png")
#     st.write("""
#     #### “Customer Segmentation” là gì?
#     - Phân khúc/nhóm/cụm khách hàng (market segmentation - còn được gọi là phân khúc thị trường) là quá trình nhóm các khách hàng lại với nhau dựa trên các đặc điểm chung. Nó phân chia và nhóm khách hàng thành các nhóm nhỏ theo đặc điểm địa lý, nhân khẩu học, tâm lý học, hành vi (geographic, demographic, psychographic, behavioral) và các đặc điểm khác.
#     - Các nhà tiếp thị sử dụng kỹ thuật này để nhắm mục tiêu khách hàng thông qua việc cá nhân hóa, khi họ muốn tung ra các chiến dịch quảng cáo, truyền thông, thiết kế một ưu đãi hoặc khuyến mãi mới, và cũng để bán hàng.
#     """)
#     st.write("""
#     #### Tại sao cần “Customer Segmentation”
#     - Để xây dựng các chiến dịch tiếp thị tốt hơn.
#     - Giữ chân nhiều khách hàng hơn: ví dụ với những khách hàng mua hàng nhiều nhất của công ty sẽ tạo ra các chính sách riêng cho họ hoặc thu hút lại những người đã mua hàng trong một khoảng thời gian.
#     - Cải tiến dịch vụ: hiểu rõ khách hàng cho phép bạn điều chỉnh và tối ưu hóa các dịch vụ của mình để đáp ứng tốt hơn nhu cầu và mong đợi của khách hàng => giúp cải thiện sự hài lòng của khách hàng.
#     - Tăng khả năng mở rộng: giúp doanh nghiệp có thể hiểu rõ hơn về những điều mà khách hàng có thể quan tâm => thúc đẩy mở rộng các sản phẩm và dịch vụ mới phù hợp với đối tượng mục tiêu của họ.
#     - Tối ưu hóa giá: Việc có thể xác định tình trạng xã hội và tài chính của khách hàng giúp doanh nghiệp dễ dàng xác định giá cả phù hợp cho sản phẩm hoặc dịch vụ của mình mà khách hàng sẽ cho là hợp lý.
#     - Tăng doanh thu: Dành ít thời gian, nguồn lực và nỗ lực tiếp thị cho các phân khúc khách hàng có lợi nhuận thấp hơn, và ngược lại. => hầu hết các phân khúc khách hàng thành công cuối cùng đều dẫn đến tăng doanh thu và lợi nhuận cũng như giảm chi phí bán hàng.
#     """)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.image('Data/onlineshop.png')
#     with col2:    
#         st.write("""
#         #### Data understanding/ requirement:
#         - Từ mục tiêu/ vấn đề đã xác định: Xem xét các dữ liệu cần thiết
#         - Toàn bộ dữ liệu được lưu trữ trong tập tin OnlineRetail.csv với 541.909 record chứa tất cả các giao dịch xảy ra từ ngày 01/12/2010 đến 09/12/2011 đối với bán lẻ trực tuyến.
#         """)
    
#     st.write("""
#     #### Attribute Information:
#     - InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
#     - StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
#     - Description: Product (item) name. Nominal.
#     - Quantity: The quantities of each product (item) per transaction. Numeric.
#     - InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
#     - UnitPrice: Unit price. Numeric, Product price per unit in sterling.
#     - CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
#     - Country: Country name. Nominal, the name of the country where each customer resides.
#     - See more at https://archive.ics.uci.edu/ml/datasets/online+retail
#     """)

# # 1. Data preparation/ Prepare===========================================================================================

# elif choice == 'Preprocessing & EDA':
#     st.subheader('II. Preprocessing & EDA')
#     st.write('#### Data Preprocessing: ')
#     st.write('##### Show data:')
    
#     st.dataframe(data.head(5))

#     st.write('#### General information: ')
#     st.write('- Number of records = ', data.shape[0])
#     st.write('- Missing values in CustomerID: ', data[data.CustomerID.isnull()].shape[0])
#     st.write('- Percentage missing values in CustomerID: ', round((data[data.CustomerID.isnull()].shape[0]) / (data.shape[0])*100, 2) ,'%')
#     st.write('- Duplicated rows: ', data.duplicated().sum())
#     st.write('- Percentage dulicated rows: ', round((data.duplicated().sum())/ (data.shape[0])*100, 2), '%')
#     st.write('- Transactions time from: ', data['InvoiceDate'].min(), 'to', data['InvoiceDate'].max())
#     st.write('- Total number customers in unique: ', data['CustomerID'].nunique())
#     st.write('- Total number transactions in unique: ', data['InvoiceNo'].nunique())
#     st.write('- Total number items in unique: ', data['StockCode'].nunique())

#     st.write('##### Data description: ')
#     st.dataframe(data.describe())

#     st.write("""visualization nummeric columns""")
#     fig = plt.figure(figsize=(12, 5))
#     sns.set_theme(style='whitegrid')
#     fig.add_subplot(121)
#     sns.distplot(data['Quantity'], kde=True)
#     fig.add_subplot(122)
#     sns.distplot(data['UnitPrice'], kde=True)
#     st.pyplot(fig)

#     st.write("""
#     **Nhận xét chung thông tin chính về tập data:**
#     - Có missing data ở cột CustomerID (khoảng 24.93%)
#     - Data bị duplicated (khoảng 1%)
#     - Cột quantity:
#         - Có giá trị âm
#         - Data range rất rộng
#     - Cột UnitPrice:
#         - Có giá trị âm
#         - Data range rất rộng
#         - Data lệch phải
#     - Cần kiểm tra những dòng có UnitPrice = 0 (khoảng 0.5% data)
#     - Cột Country: phần lớn data trong tập dữ liệu có country là UK (91.4%)
#     """)

#     # Handle Cancelled orders
#     st.write('##### Handle Cancelled invoice: ')
#     c_invoice = data[data['InvoiceNo'].astype(str).str.contains('C')]
#     st.write("""
#     **Nhận xét:**
#     - Số invoice (InvoiceNo)có 6 ký tự số, mỗi số invoice tương ứng cho mỗi lần giao dịch (transaction).
#     - Với những invoice bắt đầu bằng chữ cái 'C' đó là những canncelled invoices.
#     - Review 3 dòng đầu cancelled orders data.
#     """)
#     st.dataframe(c_invoice.head(3))
#     Total_invoice = data['InvoiceNo'].nunique()
#     Total_c_invoice = c_invoice['InvoiceNo'].nunique()

#     st.write('- Total canceled orders: ', Total_c_invoice)
#     st.write('- Percentage cancelled orders: ', round((Total_c_invoice/Total_invoice)*100, 2),'%')
#     invoices = data['InvoiceNo']
#     x = invoices.str.contains('C', regex=True)
#     x.fillna(0, inplace=True)
#     x = x.astype(int)
#     data['order_canceled'] = x
#     n1 = data['order_canceled'].value_counts()[1]
#     n2 = data.shape[0]
#     st.write('- Percentage cancelled orders vs total record: ', round((n1/n2*100),2), '%')

#     # Top 10 customer who have cancelled order
#     c_customer = pd.DataFrame(c_invoice.groupby('CustomerID')['InvoiceNo'].nunique())
#     c_customer = c_customer.sort_values(by=['InvoiceNo'], ascending=False).reset_index()
#     c_customer_10 = c_customer.head(10)
#     c_customer_10['CustomerID'] = c_customer_10['CustomerID'].astype('str')
    
#     # Visualization
#     fig = plt.figure(figsize=(7,3))
#     sns.set_theme(style='whitegrid')
#     sns.barplot(x=c_customer_10['CustomerID'], y=c_customer_10['InvoiceNo'], data=c_customer_10, palette='crest_r')
#     plt.xticks(rotation=25)
#     plt.title('Top 10 Customer cancel orders', fontsize=18, color='b')
#     st.pyplot(fig)
#     st.write("""
#     **Nhận xét:**
#     - Cancelled invoices chiếm tỷ lệ nhỏ 14.81% transactions (unique()) tương ứng 1.72% so với total record dan đầu.
#     - với những customer thường xuyên cancel invoice cần xem sét riêng để khắc phục vấn đề này.
#     - => Do đó sẽ loại bỏ cancelled invoices để phân tích. 
#     """)

#     st.write('##### Clearing data: ')
#     st.write('Number records before handle cancelled invoices, mising values, duplicated values: ', data.shape[0])
#     data = data[~(data.Quantity<0)]
#     data.dropna(subset=['CustomerID'],how='all',inplace=True)
#     data.drop_duplicates(inplace = True)
#     st.write('Number records after handle cancelled invoices, mising values, duplicated values: ', data.shape[0])

#     # Top10 countries
#     st.write('#### Select Country for analyse: ')
#     country = pd.DataFrame(data.Country.value_counts())
#     country = country.sort_values(by=['Country'], ascending=False).reset_index()
#     country['percent'] = (country['Country'] / 
#                   country['Country'].sum()) * 100
#     # Visualization
#     def with_percent(plot, feature):
#         total = sum(feature)
#         for p in ax.patches:
#             percentage = '{:.1f}%'.format(100 * p.get_height()/total)
#             x = p.get_x() + p.get_width() / 2
#             y = p.get_y() + p.get_height()
#             ax.annotate(percentage, (x, y), size = 12)
#         plt.show()

#     fig = plt.figure(figsize=(10,5))
#     sns.set_theme(style='whitegrid')
#     ax = sns.barplot(x=country['index'].head(10), y=country['Country'].head(10), data=country, palette='crest_r')
#     plt.xticks(rotation=25)
#     plt.title('Top 10 Countries in orders percentage', fontsize=18, color='b')
#     plt.xlabel('Countries', size=12)
#     plt.ylabel('Percentage', size=12)
#     with_percent(ax, country.Country)
#     st.pyplot(fig)

#     st.write("""
#     **Nhận xét:**
#     - Country có nhiều giao dịch tương ứng với số invoice nhiều nhất là United Kingdom, nên tập trung nhiều cho thị trường này với trên 80%
#     - Lựa chọn thị trường UK để phân loại và phân tích.
#     """)
    
#     # Preprae data UK
#     df_uk = data[data.Country == 'United Kingdom']
#     df_uk['TotalAmount'] = df_uk['Quantity'] * df_uk['UnitPrice']
#     st.write('Summary UK:')
    
#     st.dataframe(
#         pd.DataFrame([{'products': len(data['StockCode'].value_counts()),    
#                'transactions': len(data['InvoiceNo'].value_counts()),
#                'customers': len(data['CustomerID'].value_counts()),  
#               }], columns = ['products', 'transactions', 'customers'], index = ['quantity'])
#     )
#     # change format InvoiceDate
#     df_uk['InvoiceDate'] = pd.to_datetime(df_uk['InvoiceDate'], format='%d-%m-%Y %H:%M')
#     df_uk.InvoiceDate = pd.to_datetime(df_uk.InvoiceDate)

#     # Cohort Analysis
#     st.write('#### Customer behavior analysis in time range with Cohort: ')
#     st.write("""
#     **What is Cohort Analysis**
#     - A cohort is a set of users who share similar characteristics over time. Cohort analysis groups the users into mutually exclusive groups and their behaviour is measured over time. It can provide information about product and customer lifecycle.
#     - There are three types of cohort analysis:

#         - **Time cohorts**: It groups customers by their purchase behaviour over time.
#         - **Behaviour cohorts**: It groups customers by the product or service they signed up for.
#         - **Size cohorts**: Refers to various sizes of customers who purchase company's products or services. This categorization can be based on the amount of spending in some period of time.
#     - Understanding the needs of various cohorts can help a company design custom-made services or products for particular segments.
#         - Tài liệu tiếng Anh:https://towardsdatascience.com/a-step-by-step-introduction-to-cohort-analysis-in-python-a2cbbd8460ea
#         - Tài liệu tiếng Việt:https://200lab.io/blog/cohort-analysis-la-gi-ung-dung-phan-tich-customer-retention/
#     """)
#     # Prepare data
#     cohort_data = df_uk[['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','TotalAmount','CustomerID','Country']]
    
#     st.write('Transaction new data from: ')
#     all_dates = cohort_data.InvoiceDate.apply(lambda x:x.date())
#     st.write(' - Start date: ', all_dates.min())
#     st.write(' - End date: ', all_dates.max())
#     st.write(' - Date range: ', (all_dates.max() - all_dates.min()).days ,"days")
#     st.write('- Number of records = ', cohort_data.shape[0])

#     def get_month(x):
#         return dt.datetime(x.year, x.month, 1)

#     cohort_data['InvoiceMonth'] = cohort_data['InvoiceDate'].apply(get_month)
#     grouping = cohort_data.groupby('CustomerID')['InvoiceMonth']
#     cohort_data['CohortMonth'] = grouping.transform('min')

#     def get_date_int(df, column):    
#         year = df[column].dt.year    
#         month = df[column].dt.month    
#         day = df[column].dt.day
#         return year, month, day

#     invoice_year, invoice_month, _ = get_date_int(cohort_data, 'InvoiceMonth') 
#     cohort_year, cohort_month, _ = get_date_int(cohort_data, 'CohortMonth')
#     years_diff = invoice_year - cohort_year
#     months_diff = invoice_month - cohort_month
#     cohort_data['CohortIndex'] = years_diff * 12 + months_diff

#     grouping = cohort_data.groupby(['CohortMonth', 'CohortIndex'])
#     cohort_data_cust_id = grouping['CustomerID'].apply(pd.Series.nunique)
#     cohort_data_cust_id = cohort_data_cust_id.reset_index()
#     cohort_counts = cohort_data_cust_id.pivot(index='CohortMonth',columns='CohortIndex',values='CustomerID')
#     st.dataframe(cohort_counts)
#     st.write("""
#     **Nhận xét:** Xét CohortMonth 2010-12-01 (thời điểm bắt đầu của tập data gốc): 
#     - Với CohortIndex 0, có 815 unique customers đã thực hiện giao dịch trong tháng 12-2010. 
#     - Với CohortIndex 1, có 289 customers trong số 815 khách hàng đã thực hiện giao dịch ở CohortMonth 2010-12-01 tiếp tục thực hiện giao dịch ở tháng kế tiếp (gọi là **active customers**).
#     - Với CohortIndex 2, có 263 customers trong số 815 khách hàng đã thực hiện giao dịch ở CohortMonth 2010-12-01 tiếp tục thực hiện giao dịch ở tháng thứ 2 liền kề. 
#     - Các CohortIndex tiếp theo có logic tương tự
#     """)
#     st.write("""
#     Tính Retention Rate (%active customers trên tổng số customers)
#     """)
#     cohort_sizes = cohort_counts.iloc[:,0]
#     retention = cohort_counts.divide(cohort_sizes, axis=0)
#     st.dataframe(retention)
#     st.write("""
#     **Lưu ý:**
#     Retention rate của CohortMonth luôn bằng 100%, do số lượng active customer ở CohortMonth cũng chính bằng tổng số lượng khách hàng.
#     """)
#     # Visualization
#     fig = plt.figure(figsize=(10, 8))
#     plt.title('Retention rates')
#     sns.heatmap(data = retention,annot = True,fmt = '.0%',vmin = 0.0,vmax = 0.5, cmap = 'crest_r')
#     st.pyplot(fig)
#     st.write("""
#     **Nhận xét:**
#     Từ heatmap trên, có thể thấy:
#     - Với CohortMonth 2010-12-01, retention rate chiếm khoảng 35%. Trong đó, thời điểm có retention rate cao nhất (50%) là sau 11 tháng.
#     - Với các CohortMonth còn lại, khoảng 18%-25% khách hàng duy trì việc mua sắm tại cửa hàng.
#     """)

# # RFM Analyze & Evaluation===================================================================================================
# elif choice == 'RFM Analyze & Evaluation':
#     st.subheader('III. RFM Analyze & Evaluation')
#     st.write('#### A. RFM Segmentation: ')
#     st.write("""Show data after reprocessing: """)
#     cohort_data = pd.read_csv('Data/OnlineRetail_new.csv', encoding= 'unicode_escape')
#     cohort_data.CustomerID = cohort_data.CustomerID.astype(object)
#     cohort_data.InvoiceDate = pd.to_datetime(cohort_data.InvoiceDate)
#     st.dataframe(cohort_data.head(5))

#     st.write("""Summary information : """)
#     st.write('- Transactions time from: ', cohort_data['InvoiceDate'].min(), 'to', cohort_data['InvoiceDate'].max())
#     st.write('- Transaction have No InvoiceNo: ', cohort_data[cohort_data.InvoiceNo.isnull()].shape[0])
#     st.write('- Total number customers in unique: ', cohort_data['CustomerID'].nunique())
#     st.write('- Total number transactions in unique: ', cohort_data['InvoiceNo'].nunique())
#     st.write('- Total number items in unique: ', cohort_data['StockCode'].nunique())
#     st.dataframe(cohort_data[['TotalAmount']].describe())

#     st.write("""Transaction time selection :""")
#     st.write("""
#     - Tính toán thêm cột Recency, Frequency, Monetary dữ liệu
#     - Trước hết, cần xác định mốc thời gian chuẩn để biết được một transaction đã được thực hiện cách mốc thời gian bao lâu.
#     - Ở đây sẽ chọn mốc thời gian là max date của tập data + 1
#     - Lý do: để recency có giá trị thấp nhất = 1 thay vì = 0 và thời gian chuẩn hóa cho việc phân tích bằng tròn 1 năm.
#     """)
#     all_dates = (pd.to_datetime(cohort_data['InvoiceDate'])).apply(lambda x:x.date())
#     start_date = all_dates.max()-relativedelta(months=12,days=-1)

#     data_rfm = cohort_data[cohort_data['InvoiceDate'] >= pd.to_datetime(start_date)]
#     data_rfm.reset_index(drop=True,inplace=True)
#     snapshot_date = max(data_rfm.InvoiceDate) + dt.timedelta(days=1)

#     st.write('Transactions time selection from: ', cohort_data['InvoiceDate'].min(), 'to', cohort_data['InvoiceDate'].max())
#     st.write('Snapshot date: ', snapshot_date.date())
#     data_RFM = data_rfm.groupby(['CustomerID'],as_index=False).agg({'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
#                                              'InvoiceNo': 'count',
#                                              'TotalAmount': 'sum'}).rename(columns = {'InvoiceDate': 'Recency',
#                                                                                    'InvoiceNo': 'Frequency',
#                                                                                    'TotalAmount': 'Monetary'})
#     st.dataframe(data_RFM)
#     st.write('Summary numeric columns data:')
#     st.dataframe(data_RFM.describe())
#     data_RFM_num = data_RFM[['Recency', 'Frequency', 'Monetary']]
#     st.write('Visualization numeric columns:')
#     fig = plt.figure(figsize = (17,4))
#     for a,b in enumerate(data_RFM_num.columns):
#         ax = fig.add_subplot(1,4,a+1)
#         sns.distplot(data_RFM_num[b])
#     plt.tight_layout()
#     st.pyplot(fig)
#     st.write("""
#     - Dữ liệu lệch phải vẫn chưa được xử lý
#     - **Recency** cho thấy lần mua cuối gần nhất của khách hàng tập trung nhiều trong khoảng dưới 100 ngày ~ 3 tháng. Đường KDE này cũng có thể hình dung có 3 nhóm khách hàng
#         - Group 1: khoảng 200 ngày mới mua lại
#         - Group 2: khoảng 400 ngày mới mua hàng lại
#         - Group 3: trên 400 ngày mới mua hàng lại
#     - **Frequency** cho thấy số lượng lớn khách hàng mua hàng dưới 25 lần trong khoảng 1 năm được xét. Trung bình họ mua hàng đến 4 lần trong năm.
#     - **Monetary** cho thấy tương tự cho thấy số tiền mà khách hàng bỏ ra mua hàng trong khoảng thời gian được xét nằm trung bình trong khoảng dưới 25,000. Cụ thể 75% khách hàng đã chi 1,661. cho việc mua hàng trong năm.
#     """)
#     # Visualization
#     data_RFM['Rank'] = data_RFM['Monetary'].rank(ascending=0)
#     data_RFM = data_RFM.sort_values('Rank', ascending=True)
#     data_RFM_10 = data_RFM.reset_index().head(10)
#     data_RFM_10['CustomerID'] = data_RFM_10['CustomerID'].astype('str')
#     st.write('Top 10 Customer by Monetary: ')
#     fig = plt.figure(figsize=(7,3))
#     sns.set_theme(style='whitegrid')
#     sns.barplot(x=data_RFM_10['CustomerID'], y=data_RFM_10['Monetary'], data=data_RFM_10, palette='crest_r')
#     plt.xticks(rotation=25)
#     plt.title('Top 10 Best Customer by Monetary', fontsize=18, color='b')
#     st.pyplot(fig)

#     st.write("""
#     - Top 10 Best Customer by Monetary ranking dựa trên số tiền Customer đó bỏ ra mua hàng lớn
#     - Tuy nhiên việc đánh giá này còn dựa trên mức độ thường xuyên (Recency) mua hàng và số lần mua hàng cao (frequency)
#     """)

#     st.write("""
#     **RFM Segmentation**
#     - Tính R,F,M cho từng khách hàng CustomerID
#     - Mỗi R,F,M bao gồm 4 nhóm tạo ra 64(4,4,4) phân khúc khách hàng khác nhau thực hiện theo tứ phân vị cho từng R,F,M
#     """)
#     st.write("""Show data:""")

#     # Create new columns R, F, M
#     r_labels = range(4, 0, -1)
#     r_quartiles = pd.qcut(data_RFM['Recency'], 4, labels = r_labels)
#     data_RFM = data_RFM.assign(R = r_quartiles.values)

#     f_labels = range(1,5)
#     m_labels = range(1,5)
#     f_quartiles = pd.qcut(data_RFM['Frequency'], 4, labels = f_labels)
#     m_quartiles = pd.qcut(data_RFM['Monetary'], 4, labels = m_labels)
#     data_RFM = data_RFM.assign(F = f_quartiles.values)
#     data_RFM = data_RFM.assign(M = m_quartiles.values)
    
#     # Concat RFM quartile values to create RFM Segments
#     def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
#     data_RFM['RFM_Segment'] = data_RFM.apply(join_rfm, axis=1)

#     # Calculate RFM score and level
#     data_RFM['RFM_Score'] = data_RFM[['R','F','M']].sum(axis=1)
#     st.write("""Data after apply RFM processing :""")
#     st.dataframe(
#         data_RFM.groupby('RFM_Score').agg({'Recency': 'mean',
#                                    'Frequency': 'mean',
#                                    'Monetary': ['mean', 'count'] }).round(1)
#     )
#     st.write("""
#     Score range của RFM_Score nằm trong khoảng từ 3-12 nên sẽ tạo các segment theo logic sau:
#     - RFM_Score >= 9: khách hàng thuộc nhóm "TOP"
#     - RFM Score >= 5 và < 9: khách hàng thuộc nhóm "MIDDLE"
#     - RFM Score < 5: khách hàng thuộc nhóm "LOW"
#     """)
#     st.write("""**Manually create segment**""")
#     def create_segment(df):
#         if df['RFM_Score'] >= 9:
#             return 'TOP'
#         elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 9):
#             return 'MIDDLE'
#         else:
#             return 'LOW'
    
#     data_RFM['General_Segment'] = data_RFM.apply(create_segment, axis=1)
#     data_RFM.groupby('General_Segment').agg({'Recency': 'mean',
#                                      'Frequency': 'mean',
#                                      'Monetary': ['mean', 'count']}).round(1)  
    
#     # Calculate average values for each RFM_Level, and return a size of each segment
#     rfm_agg = data_RFM.groupby('General_Segment').agg({
#         'Recency': 'mean',
#         'Frequency': 'mean',
#         'Monetary': ['mean', 'count']}).round(0)

#     rfm_agg.columns = rfm_agg.columns.droplevel()
#     rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
#     rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

#     # Reset the index
#     rfm_agg = rfm_agg.reset_index()
#     st.dataframe(rfm_agg)
#     st.write("""
#     **Nhận xét:** Cần có những giải pháp phừ hợp với từng vấn đề sau:
#     - Nhóm TOP và MIDDLE chiếm tỷ lệ tương đương (43% vs 38%)
#     - Tuy nhiên giá trị từ TOP mang lại nhiều hơn và tần xuất mua cũng cao nhất nên cần tập trung nhóm này.
#     """)

#     # Visualization
#     fig = plt.gcf()
#     fig.set_size_inches(10, 5)

#     colors_dict = {'Low':'yellow','Middle':'royalblue', 'Top':'cyan'}

#     squarify.plot(sizes=rfm_agg['Count'],
#                 text_kwargs={'fontsize':12,'weight':'bold'},
#                 color=colors_dict.values(),
#                 label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
#                         for i in range(0, len(rfm_agg))], alpha=0.5 )

#     plt.title("Customers Segments",fontsize=20,fontweight="bold")
#     plt.axis('off')
#     st.pyplot(fig)

#     # Visualization
#     # fig = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="General_Segment",
#     #         hover_name="General_Segment", size_max=200)
#     # st.pyplot(fig)
    

#     st.write("""
#     **Nhận xét:**
#     - Việc phân chia RFM segment thành 3 nhóm Low, Medium, Top giúp phân chia tập khách hàng thành 3 nhóm với khác biệt tương đối rõ ràng. Trong đó:
#         - LOW: nhóm có số lượng ít nhất (17.56% tập data), đây là nhóm những khách hàng đã thực hiện giao dịch từ rất lâu, số lượng và giá trị giao dịch thực hiện đều khá thấp
#         - MIDDLE: nhóm đông nhất (43.68%), đây là nhóm với tất cả các chỉ số (recency, frequency và moneytary) đều ở mức trung bình.
#         - TOP: nhóm chiếm 38.75% tập data nhưng đem lại lợi nhuận lớn nhất, thực hiện nhiều giao dịch nhất, lần cuối mua hàng cũng là gần đây nhất
#     """)
#     # K-means Clustering==================================================================================================
#     st.write('#### B. K-means Clustering: ')
#     st.write('Review data: ')
#     g = sns.pairplot(data_RFM, hue='General_Segment', diag_kind='kde', vars=['Recency', 'Frequency', 'Monetary'])
#     g.map_lower(sns.regplot)
#     st.pyplot(g)

#     col1, col2 = st.columns(2)
#     with col2:
#         st.write("""
#         - Dữ liệu lệch phải và chưa trực quan hóa rõ ràng các nhóm cần xử lý.
#         - Trước khi xử lý cần tính phương sai
#         - phương sai như kết quả bên cao nên sẽ thực hiện log để giảm phương sai và nhóm dữ liệu thể hiện rõ hơn.
#         """)
#     with col1:
#         st.write("""**Chuẩn hóa dữ liệu:**""")
#         st.dataframe(
#             pd.Series(data_RFM[['Recency', 'Frequency', 'Monetary']].var(), name="Variance")
#         )
#     st.write("""
#     - Xử lý khách hàng có giá trị mua hàng = 0 là không hợp lý.
#     """)
#     st.dataframe(data_RFM[data_RFM['Monetary'] == 0])
#     data_RFM = data_RFM[data_RFM['Monetary'] > 0]
#     data_RFM.reset_index(drop=True,inplace=True)

#     # Chuẩn hóa dữ liệu
#     df = data_RFM[['Recency','Frequency','Monetary']]
#     df_log = np.log(df)
#     scaler = StandardScaler()
#     scaler.fit(df_log)
#     data_normalized = scaler.transform(df_log)
#     df_norm = pd.DataFrame(data=df_log, index=df.index, columns=df.columns)

#     # Visualization:
#     st.write('**Dữ liệu sau khi chuẩn hóa**')
#     fig = plt.figure(figsize = (17,4))
#     for a,b in enumerate(df_norm.columns):
#         ax = fig.add_subplot(1,3,a+1)
#         sns.distplot(df_norm[b])
#     plt.tight_layout()
#     st.pyplot(fig)
#     st.write("""**Nhận xét:** Sau khi áp dụng log/scale transformation đã xử lý được việc lệch phải của data""")

#     st.write("""**Build K-mean:**""")
#     df_kmean = df_norm.copy(deep=True)
#     from sklearn.cluster import KMeans
#     sse = {}
#     for k in range(1, 11):
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(df_kmean)
#         sse[k] = kmeans.inertia_
#     key = list(sse.keys())
#     sse_value = list(sse.values())
#     zipped = list(zip(key, sse_value))
#     # Visualization
#     fig = plt.figure(figsize=(5,4))
#     plt.title('The Elbow Method')
#     plt.xlabel('k')
#     plt.ylabel('SSE')
#     sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
#     # Show chart
#     col1, col2 = st.columns(2)
#     with col2:
#         st.dataframe(pd.DataFrame(zipped, columns=['K', 'WSSE']))
#     with col1:
#         st.pyplot(fig)
#     st.write("""
#     **Nhận xét:** Chọn k=3 hoặc k=4 khoảng cách giữa các cluster ổn định gần với tập dữ liệu
#     """)
#     # k=3
#     k = 3
#     model = KMeans(n_clusters=5, random_state=42)
#     model.fit(df_kmean)
#     kmeans = KMeans(n_clusters=3, random_state=42)
#     kmeans.fit(df_kmean)
#     cluster_labels = kmeans.labels_
#     df_kmean_k3 = df_kmean.assign(Cluster = cluster_labels)
#     df_k3 = df.assign(Cluster = cluster_labels)
#     summary_k3 = df_k3.groupby(['Cluster']).agg({'Recency': 'mean',
#                                                         'Frequency': 'mean',
#                                                         'Monetary': ['mean', 'count'],}).round(0)

#     df_kmean_k3.index = data_RFM['CustomerID'].astype(int)
#     data_melt3 = pd.melt(df_kmean_k3.reset_index(),
#                         id_vars=['CustomerID', 'Cluster'],
#                         value_vars=['Recency', 'Frequency', 'Monetary'],
#                         var_name='Attribute',
#                         value_name='Value')
#     # k=4
#     kmeans = KMeans(n_clusters=4, random_state=42)
#     kmeans.fit(df_kmean)
#     cluster_labels = kmeans.labels_
#     df_kmean_k4 = df_kmean.assign(Cluster = cluster_labels)
#     df_k4 = df.assign(Cluster = cluster_labels)
#     summary_k4 = df_k4.groupby(['Cluster']).agg({'Recency': 'mean',
#                                                         'Frequency': 'mean',
#                                                         'Monetary': ['mean', 'count'],}).round(0)

#     df_kmean_k4.index = data_RFM['CustomerID'].astype(int)
#     data_melt4 = pd.melt(df_kmean_k4.reset_index(),
#                         id_vars=['CustomerID', 'Cluster'],
#                         value_vars=['Recency', 'Frequency', 'Monetary'],
#                         var_name='Attribute',
#                         value_name='Value')

#     col1, col2 = st.columns(2)
#     with col1:
#         st.write('**k = 3**')
#         st.dataframe(summary_k3)
#     with col2:
#         st.write('**k = 4**')
#         st.dataframe(summary_k4)

#     st.write('Visualization the result')
#     fig = plt.figure(figsize=(15,4))
#     sns.set_style("white")
#     plt.subplot(1,2,1)
#     sns.lineplot(x="Attribute", y="Value", hue='Cluster', data=data_melt3,
#                 palette=sns.color_palette('muted',n_colors=3)).set_title("k = 3")
#     plt.subplot(1,2,2)
#     sns.lineplot(x="Attribute", y="Value", hue='Cluster', data=data_melt4,
#                 palette=sns.color_palette('muted',n_colors=4)).set_title("k = 4")
#     st.pyplot(fig)
#     st.write("""
#     **Kết luận:**
#     *(Lưu ý: giá trị recency càng thấp tức khách hàng vừa mua hàng gần đây và ngược lại)*
#     - Có thể thấy giữa k = 3 và 4 có 3 nhóm cluster có tính chất tương tự nhau.
#     - Với k = 4, tập data được phân nhóm cụ thể hơn, trong đó xuất hiện nhóm cluster 3 như một bước chuyển tiếp từ nhóm khách hàng bình thường (R,F,M trung bình) sang nhóm khách hàng có R,F,M xấu
#     - => Quyết định chọn k = 4
#     """)

#     # Hierarchical Clustering==================================================================================================
#     st.write('#### B. Hierarchical Clustering: ')
#     st.write('*(Hierarchical Clustering phù hợp với nhiều kiểu dữ liệu nên sẽ ưu tiên sử dụng)*')
#     # Dendrogram
#     df_H = df_norm.copy(deep=True)

#     import scipy.cluster.hierarchy as shc
#     fig = plt.figure(figsize=(15,5))
#     plt.title('Customer Dendograms')
#     dend = shc.dendrogram(shc.linkage(df_H, method='complete'))
#     st.pyplot(fig)
#     st.write('**Nhận xét:** Lựa chọn cluster = 4 do có tỷ lệ mẫu khá đồng đều nhất')
#     group = AgglomerativeClustering(n_clusters=4, affinity='euclidean')
#     group.fit(df_H)
#     df_H['Group'] = group.labels_.tolist()
#     st.write('**Cluster = 4**')
#     summary_H = df_H.groupby(['Group']).agg({'Recency': 'mean',
#                                                     'Frequency': 'mean',
#                                                     'Monetary': ['mean', 'count'],}).round(0)
#     df_H.index = data_RFM['CustomerID'].astype(int)
#     df_H_melt = pd.melt(df_H.reset_index(),
#                         id_vars=['CustomerID', 'Group'],
#                         value_vars=['Recency', 'Frequency', 'Monetary'],
#                         var_name='Attribute',
#                         value_name='Value')
#     # Compare K-mean va H
#     st.write('**Compare K-mean vs Hierarchical Clustering**')
#     fig = plt.figure(figsize=(15,4))
#     sns.set_style("white")
#     plt.subplot(1,2,1)
#     sns.lineplot(x="Attribute", y="Value", hue='Cluster', data=data_melt4,
#                 palette=sns.color_palette('muted',n_colors=4)).set_title("K-mean")
#     plt.subplot(1,2,2)
#     sns.lineplot(x="Attribute", y="Value", hue='Group', data=df_H_melt,
#              palette=sns.color_palette('muted',n_colors=4)).set_title("Hierarchical")
#     plt.show()
#     st.pyplot(fig)
#     st.write("""
#     **Kết luận:** 
#     - lựa chọn giải pháp K-mean thấy rõ được các nhóm hơn
#     - Với Hierarchical khả năng attribute nhóm 1 và 3 là như nhau.
#     """)
# # Conclusion and Suggestion==================================================
# elif choice == 'Conclusion and Suggestion':
#     st.subheader('IV. Conclusion and Suggestion')
#     st.write('Average values by clusters after K-mean:')
#     df_k4 = pd.read_csv('Data/OnlineRetail_k4.csv', encoding= 'unicode_escape')
#     cluster_avg = df_k4.groupby(['Cluster']).mean()

#     # Calculate average values for each RFM_Level, and return a size of each segment 
#     rfm_agg2 = df_k4.groupby('Cluster').agg({
#         'Recency': 'mean',
#         'Frequency': 'mean',
#         'Monetary': ['mean', 'count']}).round(1)

#     rfm_agg2.columns = rfm_agg2.columns.droplevel()
#     rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
#     rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

#     # Reset the index
#     rfm_agg2 = rfm_agg2.reset_index()

#     # Change thr Cluster Columns Datatype into discrete values
#     rfm_agg2['Cluster'] = 'Cluster '+ rfm_agg2['Cluster'].astype('str')
#     st.dataframe(rfm_agg2)
#     st.image("Data/k4.png")
#     st.write("""
#     Với k = 4, kết hợp với phân tích RFM & Cohort, có thể tóm tắt các đặc điểm của 4 nhóm khách hàng và các gợi ý mà doanh nghiệp có thể tham khảo để cải thiện hoạt động kinh doanh
#     """)
#     st.write("""
#     | Cluster| Customer types|                RFM                    |        Recommendation Action
#     |--------|---------------|---------------------------------------|------------------------------------------|
#     |    0   |     Lost      |Khách hàng chi tiêu thấp nhất ít thường xuyên nhất và đã lâu chưa quay lại mua hàng|Tìm hiểu lý do nhóm này rời bỏ doanh nghiệp bằng cách phân tích data cũ (hàng đã mua, giá trị, chất lượng...)|
#     |    1   |     New       |Khách hàng vừa mua hàng gần đây vì là khách mới nên frequency thấp và chi tiêu tương đối thấp|Khảo sát để hiểu rõ trải nghiệm khách hàng. có thể dùng làm tư liệu phân tích khách hàng chuyển biến trong tương lai|
#     |    2   |     Best      |Khách hàng chi tiêu thường xuyên nhất với số tiền chi tiêu cao nhất và vừa thực hiện giao dịch gần đây  |Khách hàng chiến lược của công ty cần được ưu tiên chăm sóc mới|
#     |    3   |     Risk      |Khách hàng đã lâu chưa quay lại cũng thường xuyên mua với giá trị trung bình |Nguy cơ khách hàng rời bỏ doanh nghiệp là rất cao, cần thực hiện các chương trình giảm giá thu hút khách hàng|
#     """)

#     st.write("""
#     **Nhận xét chung:**
#     Cohort analysis cho thấy xu hướng khách hàng rời bỏ doanh nghiệp trong thời gian gần đây có xu hướng tăng, doanh nghiệp cần so sánh chỉ tiêu này với số liệu của trung bình ngành để tự đánh giá mức độ khách hàng rời bỏ hiện tại là hợp lý hay cần phải cải thiện
#     """)
#     st.subheader('Kết Luận:')
#     st.write("""
#     - Nếu nguồn lực của doanh nghiệp có hạn và cẩn sắp xếp theo thứ tự ưu tiên các nhóm khách hàng cần chăm sóc, có thể xem xét thự tự gợi ý sau (thứ tự này được phân tích dựa trên Domain Knowledge kết hợp với Monetary Value mà nhóm khách hàng mang lại cho doanh nghiệp):
#     1. **Nhóm Best**: đây là nhóm ổn định, mang lại lợi nhuận cao cho doanh nghiệp nên cần được ưu tiên chăm sóc và tư vấn sản phẩm, giới thiệu sản phẩm mới và các sản phẩm độc quyền. Tuy nhiên, vì đây là nhóm đã ổn định và có chi tiêu cao nên không cần tập trung chạy các chương trình khuyến mãi giảm giá để thu hút.

#     2. **Nhóm Risk**: khả năng rất cao khách hàng ở nhóm này sẽ chuyển sang nhóm Lost, doanh nghiệp cần tung nhiều chương trình khuyến mãi để thu hút và giữ chân khách hàng, đồng thời nghiên cứu lý do khách hàng không còn hứng thú nhiều với doanh nghiệp thông qua khảo sát. Nhóm này khá đông (~29.76%) và cũng đem lại lợi nhuận tương đối lớn cho doanh nghiệp.

#     3. **Nhóm New**: đây là nhóm khách hàng mới, việc họ sẽ chuyển thành nhóm Risk hoặc nhóm Best trong tương lai phụ thuộc rất lớn vào trải nghiệm mua sắm và các chương trình chăm sóc, khuyến mãi của doanh nghiệp nên cần được ưu tiên.

#     4. **Nhóm Lost**: đây là nhóm khách hàng đã mất đi và gần như không có khả năng phục hồi, doanh nghiệp có thể duy trì mối quan hệ với khách hàng bằng cách gửi email/SMS thông tin các chương trình, không nên tập trung quảng cáo/khuyến mãi cho nhóm này

#     - Nhìn chung, nguồn lực của doanh nghiệp là hữu hạn, do đó các chương trình nên tập trung vào nhóm khách hàng mang lại lợi nhuận nhiều nhất, đó là nhóm Best và Risk.
#     """)
  
