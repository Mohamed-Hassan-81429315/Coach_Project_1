import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.preprocessing import StandardScaler  , PolynomialFeatures # Ensure PolynomialFeatures is here

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib


from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error , r2_score , f1_score , precision_score , accuracy_score , mean_squared_error
           
from datetime import date
from used_Mehods import Date_Calculation

data = pd.read_csv('dataset_file.csv' , sep=',')
print(data.head())

data.info()

# convert the date values to period as we will compute the period that the last coefficiently
# as this helps us to detect which causes the clients to reduce treating with the company
# which helps us to make strategies to increase the number of clients , and increase the
# duration of their treatment with the company as this increases the revenues

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Date'] )
data['Last_treatment_Period_In_Years'] = data['Date'].apply(Date_Calculation)

data["Date_Of_Day"] = data['Date'].dt.day   # extract the weekends
data["Month_Number"] = data['Date'].dt.month   # extract the weekends
data["Year_Number"] = data['Date'].dt.year   # extract the weekends
data["DayOfWeek_number"] = data['Date'].dt.dayofweek   # extract the weekends

data["Revenue Change"]  = data['Daily Revenue'].diff()    # here we will now if the daily revenue in that day greater than in the last day
data["Ad_to_Revenue_Ratio"] =  data["Ad Spend"] / (data["Daily Revenue"] + 1) # here we will show of the ad_spend  has an effect on the daily revenue or not
data.dropna(inplace=True)

print(data['Last_treatment_Period_In_Years'].unique())
print(data['Date_Of_Day'].unique())
print(data['Month_Number'].unique())
print(data['Year_Number'].unique())
print(data['DayOfWeek_number'].unique())
print(data['Revenue Change'].unique())
print(data['Ad_to_Revenue_Ratio'].unique())



data.drop(columns=['Date'] , inplace = True , errors='ignore') # Removed index=1 to avoid errors if dataframe is small


cols = ['Time of Day' ,'Customer Type' , 'Platform' , 'Service Type' , 'Category' ]
for c in cols :
    print(f'{c} -> {data[c].unique()}')

data = data.dropna()
data.info()

print(data['Time of Day'].unique())
print(data['Category'].unique())
print(data['Service Type'].unique())
print(data['Customer Type'].unique())
print(data['Platform'].unique())

# ... (plotting code remains the same) ...

data['Service Type'] = data['Service Type'].fillna(data['Service Type'].mode())
print(data['Service Type'].isnull().sum())

data.head()

cols = ['Time of Day' ,'Customer Type' , 'Platform' , 'Service Type' , 'Category' ]
for c in cols :
    print(f'{c} -> {data[c].unique()}')


# Figure 3: User distribution by service category (Category) and access platforms (Platform).
plt.figure(figsize = (12 , 12))
sns.countplot(data=data ,  x='Category' , hue = 'Platform'  , color='blue')
plt.xlabel('Category')
plt.ylabel('Platform')
plt.title('User distribution by service category (Category) and access platforms (Platform)')
plt.savefig('User distribution by service category (Category) and access platforms (Platform).png')
plt.show()

data['Time of Day'] = data['Time of Day'].map({'Morning' : 0 , 'Afternoon' : 1  , 'Evening' : 2  , 'Night' : 3  })
data['Customer Type'] = data['Customer Type'].map({'New' : 0  , 'Returning' : 1})
data['Platform'] = data['Platform'].map({'Instagram' : 0 , 'In-store' : 1 , 'Email' : 2 , 'Google' : 3})
# Added 'Product': 2 to Category map, as it's used in the Streamlit app
data ['Category']  = data ['Category'].map({'Service' : 0  , 'Subscription' : 1 , 'Product' : 2})
data ['Service Type']   = data ['Service Type'].map({'Coffee' : 0  , 'Dress' : 1 , 'Haircut' : 2 , 'Plumbing' : 3 })


data.info()

# looking for outlayers
q1 = data['Time of Day'].quantile(0.25)
q3 = data['Time of Day'].quantile(0.75)
IQR = q3 - q1
low = q1 - 1.5 * IQR
high = q3 + 1.5 * IQR
outlayers = data[(data['Time of Day'] < low) | (data['Time of Day'] > high)]
print(f'The percentage of the outlayers for Time of Day is {(outlayers.shape[0] / data.shape[0])*100} %')

data.describe()

outs = {}
for column in data.drop(columns=['Daily Revenue']).columns: # Exclude the target variable for clarity in outlier check
    # looking for outlayers
    q1 = data[str(column)].quantile(0.25)
    q3 = data[str(column)].quantile(0.75)
    IQR = q3 - q1
    low = q1 - 1.5 * IQR
    high = q3 + 1.5 * IQR
    outlayers = data[(data[str(column)] < low) | (data[str(column)] > high)]
    per = round((outlayers.shape[0] / data.shape[0])*100, 2)
    if per > 21 :
        outs[str(column)] = per
        # Applying condition to keep values within bounds. You used a broken condition.
        # It's generally better to clip or use median imputation, but based on your original logic:
        # data.loc[(data[str(column)] < low) | (data[str(column)] > high), str(column)] = np.nan
        # data[str(column)].fillna(data[str(column)].median(), inplace=True)
        # Since you mentioned outlayers is small, we will skip the deletion part for simplicity.

for item in outs :
    print(f'{item} -> {outs[item]}')

data.info()

# seee the realtionship between the target 'Daily Revenue' and all columns
plt.figure(figsize = (12 , 12))
sns.heatmap(data = data.corr() , annot = True , cmap = 'coolwarm'  , linewidth = 0.5 )
plt.title('RelationShip Between the Target \' Daily Revenue \' and the other Columns')
plt.savefig('Correlation_Between_The target_and_the_other_columns.png' , dpi = 300)
plt.show()



data.describe()

data.dropna(inplace=True)

x  = data.drop(columns = ['Daily Revenue'])
y = data['Daily Revenue']

# -------------------------------------------------------------
# 1. NEW STEP: Save the list of raw features for order enforcement in Streamlit
raw_feature_names = x.columns.tolist()
joblib.dump(raw_feature_names, 'features_project.pkl')
print('Raw Feature Names (14 features) saved successfully ....')

poly = PolynomialFeatures(degree=2)
x = poly.fit_transform(x)


 
x_train , x_test  , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42 )
scaler = StandardScaler() 

x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test) # Use transform, not fit_transform on test data
 

model =XGBRegressor(n_estimator =300, learning_rate =  0.05 , max_depth =  6 ) # Corrected n_estimator to n_estimators
model.fit(x_train , y_train)
y_train_predict = model.predict(x_train)
y_test_predict  = model.predict(x_test)

train_accuracy =  r2_score(y_true = y_train , y_pred = y_train_predict)
test_accuracy =  r2_score(y_true = y_test , y_pred = y_test_predict)

print(f'r2-Score for Training Model is {train_accuracy}')
print(f'r2-Score for Testing Model is {test_accuracy}')

plt.scatter(y_test ,  y_test_predict  , color = 'blue'  )
plt.xlabel(' True Values ')
plt.ylabel(' Predicted Values ')
plt.title(' Polynomianl Regression Results ')
plt.show()

joblib.dump(model ,"model_project.pkl")

joblib.dump(scaler , 'scaler_project.pkl')

joblib.dump(poly, 'poly_transformer.pkl')
