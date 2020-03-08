import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

from statsmodels.imputation.mice import MICE
import statsmodels.api as sm
import pandas
from statsmodels.imputation import mice
import pickle
warnings.filterwarnings("ignore")

#set to unlimited colums display
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

def Modeling(data,test_data,label):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    #from sklearn.svm import SVC
    #clf = SVC(gamma='auto')
    clf.fit(data_scaled, label)
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    y_predicted = clf.predict_proba(test_data)[:, 0]
    #df_filled = pd.DataFrame(data=y_predicted, columns=['wrf1','wtf2'])
    res_df = pd.DataFrame({'id': _id, 'label': y_predicted})
    res_df.to_csv("submission.csv",index=False)

if __name__ == '__main__':
    train_df = pd.read_csv('data_train.csv')
    test_df = pd.read_csv('data_test.csv')
    label =  pd.read_csv('data_train_label.csv',names=['label'], header=None)
    _id = pd.read_csv("data/test.csv")
    _id = _id.id
    Modeling(train_df,test_df,label)
