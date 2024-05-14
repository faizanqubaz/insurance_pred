import pandas as pd
from distribution import check_distribution
from encode import encode_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Read the Data_set
data = pd.read_csv(r"C:/Users/CL/Desktop/insurance.csv")

# CHECK THE DATA SIZE
print('size',data.shape)

# CHECK THE DATA
print('data',data.head())

# check the data types
print('dtypes',data.dtypes)

# CHECK THE MATHEMATICALL
print(data.describe())


# CHECK THE NULL VALUES
print(data.isnull().sum())


# CHECK THE Dublicated values
print(data.duplicated())


# CHECK FOR THE CORR
print(data.corr()['charges'])


# check_distribution

# check_distribution(data)

categorical_columns = [col for col in data if data[col].dtypes == 'object' ]




encode=encode_data(data[categorical_columns])
print(encode.shape)

data.drop(columns=categorical_columns,inplace=True)

combined = pd.concat([encode.reset_index(drop=True),data.reset_index(drop=True)],axis=1)

print(combined)


X_train,X_test,Y_train,Y_test = train_test_split(combined.iloc[:,0:8],combined.iloc[:,-1],test_size=0.2,random_state=9)

lr = LinearRegression()

lr.fit(X_train,Y_train)

y_pred = lr.predict(X_test)

# CHECK THE ACCURACY
score = r2_score(Y_test,y_pred)
print(score)
