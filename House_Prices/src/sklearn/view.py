import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open('./db/submission/pre/data_pre_processed_test.pkl', mode='rb') as f:
    X_test_processed, X_test_processed_pca = pk.load(f)

IDs = pd.read_csv('./db/submission/datas/IDs.csv', sep=',')
model = pk.load(open('./db/submission/models/no_pca_svr_model.pkl', mode='rb'))

prev = np.expm1(model.predict(X_test_processed))

submission = {
    'Id': IDs['Id'],
    'SalePrice': prev
}

submission_df = pd.DataFrame(submission)

submission_df.to_csv('./db/submission/datas/submission.csv', sep=',', index=False)



