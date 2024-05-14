from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import pandas  as pd


def encode_data(df):
    ct = ColumnTransformer([
        ('ohe',OneHotEncoder(drop='first',sparse_output=False),['sex','smoker','region'])
    ])

    ct_trans = ct.fit_transform(df)

    ct_columns = ct.get_feature_names_out()

    data = pd.DataFrame(ct_trans,columns=ct_columns)
    return data
