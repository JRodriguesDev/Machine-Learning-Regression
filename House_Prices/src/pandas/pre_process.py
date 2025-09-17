import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

data_train = pd.read_csv('./db/datas/train.csv', sep=',')
data_test = pd.read_csv('./db/datas/test.csv', sep=',')

dataY = data_train['SalePrice']
dataX = data_train.drop(['Id', 'SalePrice'], axis=1)
IDs = data_test['Id']
data_test = data_test.iloc[:, 1:]

datas = [dataX, data_test]
list_nan_numeric_cols = []

for i, data in enumerate(datas):
    data = data.replace('-', np.nan)
    data = data.replace('', np.nan)

    numeric_cols = data.select_dtypes(include=np.number).columns.to_list()
    object_cols = data.select_dtypes(include='object').columns.to_list()

    for col in object_cols:
        data[col] = data[col].astype('string').astype('category')

    for col in numeric_cols:
        data.loc[data[col] < 0, col] = np.nan

    categoric_cols = data.select_dtypes(include='category').columns.to_list()
    nan_categoric_cols = data[categoric_cols].columns[data[categoric_cols].isnull().any().tolist()]

    for col in nan_categoric_cols:
        data[col] = data[col].astype('string')
        data[col] = data[col].fillna('None')
        data[col] = data[col].astype('category')

    nan_numeric_cols = data[numeric_cols].columns[data[numeric_cols].isnull().any()].tolist()
    list_nan_numeric_cols.append(nan_numeric_cols)
    datas[i] = data

list_medians = []

for col in list_nan_numeric_cols[0]:
    list_medians.append({
        col: datas[0][col].median()
    })

eletrica_mode = datas[0]['Electrical'].mode()[0]

cols_to_fillna_zero_before_int = [
    'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'YearBuilt', 'YearRemodAdd',
    'GarageArea', 'GarageCars'
]


for idx, data in enumerate(datas):
    for inter in list_medians:
        col, value = list(inter.items())[0]
        data[col] = data[col].fillna(value)

    for col_to_fill in cols_to_fillna_zero_before_int:
        if col_to_fill in data.columns and data[col_to_fill].isnull().any():
            data[col_to_fill] = data[col_to_fill].fillna(0) # Preenche NaNs com 0 para estas colunas

    data['Electrical'] = data['Electrical'].fillna(eletrica_mode)
    data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF'].astype('int64')
    data['TotalBath'] = data['FullBath'] + (data['HalfBath'] * 0.5) + data['BsmtFullBath'] + (data['BsmtHalfBath'] * 0.5)
    data['AgeAtSale'] = data['YrSold'] - data['YearBuilt'].astype('int64')
    data['YearsSinceRemodel'] = data['YrSold'] - data['YearRemodAdd'].astype('int64')

    datas[idx] = data

dataX, data_test = datas

ordinal_cols_1 = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'BsmtQual', 'BsmtCond']
ordinal_categories_1 = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
ordinal_cols_2 = ['BsmtExposure']
ordinal_categories_2 = ['None', 'No', 'Mn', 'Av', 'Gd']
ordinal_cols_3 = ['BsmtFinType1', 'BsmtFinType2']
ordinal_categories_3 = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
ordinal_cols_4 = ['Fence']
ordinal_categories_4 = ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']

all_ordinal_cols = ordinal_cols_1 + ordinal_cols_2 + ordinal_cols_3 + ordinal_cols_4

all_ordinal_categories = []

for _ in ordinal_cols_1:
    all_ordinal_categories.append(ordinal_categories_1)

for _ in ordinal_cols_2:
    all_ordinal_categories.append(ordinal_categories_2)

for _ in ordinal_cols_3:
    all_ordinal_categories.append(ordinal_categories_3)

for _ in ordinal_cols_4:
    all_ordinal_categories.append(ordinal_categories_4)

#X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size= 0.25, shuffle=True)

numeric_cols_final = dataX.select_dtypes(include=np.number).columns.to_list()
categoric_cols_final = dataX.select_dtypes(include='category').columns.to_list()
nominal_cols = [col for col in categoric_cols_final if col not in all_ordinal_cols]

standard = StandardScaler()
O_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Ordinal_encoder = OrdinalEncoder(categories=all_ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1)
pca = PCA(n_components=0.95)

X_train_categoric_cols = O_encoder.fit_transform(dataX[nominal_cols])
X_train_numeric_cols = standard.fit_transform(dataX[numeric_cols_final])
X_train_ordinal_cols = Ordinal_encoder.fit_transform(dataX[all_ordinal_cols])
features_O_cols = O_encoder.get_feature_names_out(nominal_cols)

X_test_categoric_cols = O_encoder.transform(data_test[nominal_cols])
X_test_numeric_cols = standard.transform(data_test[numeric_cols_final])
X_test_ordinal_cols = Ordinal_encoder.transform(data_test[all_ordinal_cols])

X_train_categoric_df = pd.DataFrame(X_train_categoric_cols, columns=features_O_cols, index=dataX.index)
X_train_numeric_df = pd.DataFrame(X_train_numeric_cols, columns=numeric_cols_final, index=dataX.index)
X_train_ordinal_df = pd.DataFrame(X_train_ordinal_cols, columns=all_ordinal_cols, index=dataX.index)

X_test_categoric_df = pd.DataFrame(X_test_categoric_cols, columns=features_O_cols, index=data_test.index)
X_test_numeric_df = pd.DataFrame(X_test_numeric_cols, columns=numeric_cols_final, index=data_test.index)
X_test_ordinal_df = pd.DataFrame(X_test_ordinal_cols, columns=all_ordinal_cols, index=data_test.index)

X_train_processed = pd.concat([X_train_categoric_df, X_train_numeric_df, X_train_ordinal_df], axis=1)
X_test_processed = pd.concat([X_test_categoric_df, X_test_numeric_df, X_test_ordinal_df], axis=1)

X_train_processed_pca = pca.fit_transform(X_train_processed)
X_test_processed_pca = pca.transform(X_test_processed)

Y_train = np.log1p(dataY)
#Y_test = np.log1p(Y_test)

with open('./db/submission/pre/data_pre_processed_train.pkl', mode='wb') as f:
    pk.dump([X_train_processed, X_train_processed_pca, Y_train], f)

with open('./db/submission/pre/data_pre_processed_test.pkl', mode='wb') as f:
    pk.dump([X_test_processed, X_test_processed_pca], f)

IDs.to_csv('db/submission/datas/IDs.csv', sep=',', index=False)



