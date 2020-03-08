import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

from statsmodels.imputation.mice import MICE
import statsmodels.api as sm 
import pandas
pandas.options.mode.use_inf_as_na = True
from statsmodels.imputation import mice
import pickle
warnings.filterwarnings("ignore")

#set to unlimited colums display
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

class eda_n_cleaning(object):
    
    def __init__(self, data_path):
        self.data_path = data_path
    
    def read_data(self):
        self.df = pd.read_csv(self.data_path)
        print("Data's shape = ",self.df.shape)

        self.cat_features = ['province', 'district', 'maCv',
                'FIELD_7', 'FIELD_8', 'FIELD_9',
                'FIELD_10', 'FIELD_13', 'FIELD_17', 
                'FIELD_24', 'FIELD_35', 'FIELD_39', 
                'FIELD_41', 'FIELD_42', 'FIELD_43', 
                'FIELD_44']
        self.bool_features = ['FIELD_2', 'FIELD_18', 'FIELD_19', 
                 'FIELD_20', 'FIELD_23', 'FIELD_25', 
                 'FIELD_26', 'FIELD_27', 'FIELD_28', 
                 'FIELD_29', 'FIELD_30', 'FIELD_31', 
                 'FIELD_36', 'FIELD_37', 'FIELD_38', 
                 'FIELD_47', 'FIELD_48', 'FIELD_49']

        self.num_features = [col for col in self.df.columns if col not in self.cat_features + self.bool_features]
        self.cat_data = self.df[self.cat_features]
        self.boolean_data = self.df[self.bool_features]
        self.num_data = self.df[self.num_features]
        try:
            self.label = self.num_data['label']
            self.num_data = self.num_data.drop(["label"],axis=1)
        except KeyError:
            self.label = None
        return self.num_data,self.cat_data,self.boolean_data,self.label

    def eda(self):
        self.num_data,self.cat_data,self.boolean_data,self.label = self.read_data()
        # Numerical Data
        self.num_data = self.df[self.num_features]
        #corr = num_data.corr()
        #corr.style.background_gradient(cmap='coolwarm').set_precision(2)
        
        """
        **Handling these columns which have high correlation -> avoid multicollinearity**
        - age_source1 and age_source2
        - FIELD_56 and FIELD_57
        - FIELD_52 and FIELD_53
        """

        # AGE SOURCE
        # Take the mean between age_source1 and age_source2
        self.temp = []
        for i,j in zip(self.num_data["age_source1"],self.num_data["age_source2"]):
            if i is not None and j is not None:
                self.temp.append((i+j)/2)
        self.num_data["age_source"] = self.num_data["age_source2"].combine_first(self.num_data["age_source1"])

        for k,l in enumerate(self.temp):
            if (l >0 ) and (isinstance(l,float) == True) and k <10:
                self.num_data["age_source"][k] = l
        self.num_data = self.num_data.drop(["age_source1","age_source2"], axis=1)
        
        for i in self.num_data[self.num_data["age_source"] < 18]['id']:
                self.num_data["age_source"][i] = self.num_data.age_source.head()[0] #**


        # ### FIELD_56 n FIELD_57 
        # Take the mean between FIELD_56 and FIELD_57
        self.temp = []
        for i,j in zip(self.num_data["FIELD_56"],self.num_data["FIELD_57"]):
            if i is not None and j is not None:
                self.temp.append((i+j)/2)
        self.num_data["FIELD_567"] = self.num_data["FIELD_56"].combine_first(self.num_data["FIELD_57"])
        
        for k,l in enumerate(self.temp):
            if (l >0 ) and (isinstance(l,float) == True) and k <10:
                self.num_data["FIELD_567"][k] = l
        self.num_data = self.num_data.drop(["FIELD_56","FIELD_57"], axis=1)

        ### FIELD_52 n FIELD_53
        # Take the mean between FIELD_56 and FIELD_57
        temp = []
        for i,j in zip(self.num_data["FIELD_52"],self.num_data["FIELD_53"]):
            if i is not None and j is not None:
                self.temp.append((i+j)/2)
        self.num_data["FIELD_523"] = self.num_data["FIELD_52"].combine_first(self.num_data["FIELD_53"])
        for k,l in enumerate(self.temp):
            if (l >0 ) and (isinstance(l,float) == True) and k <10:
                self.num_data["FIELD_523"][k] = l
        self.num_data = self.num_data.drop(["FIELD_52","FIELD_53"], axis=1)
        
        
        # Remove ID
        self.num_data = self.num_data.drop(['id'],axis=1)
        
        #Mixed Data
        def convert_to_int(col):
            self.temp = []
            for i in col:
                try:
                    if i.isnumeric() ==True:
                        self.temp.append(int(i))
                    else:
                        self.temp.append(np.float64(np.nan))
                except AttributeError:
                    self.temp.append(np.float64(np.nan))
            return self.temp
        self.num_data["FIELD_11"] = convert_to_int(self.num_data["FIELD_11"])

        ##FIELD_12
        self.num_data["FIELD_12"] = self.num_data["FIELD_12"].replace("HT",'0')
        self.num_data["FIELD_12"] = self.num_data["FIELD_12"].replace("TN",'1')
        self.num_data["FIELD_12"] = convert_to_int(self.num_data["FIELD_12"])
        
        ##FIELD_40
        self.num_data["FIELD_40"] = self.num_data["FIELD_40"].replace("08 02",'999')
        self.num_data["FIELD_40"] = self.num_data["FIELD_40"].replace("05 08 11 02",'999')
        self.num_data["FIELD_40"] = self.num_data["FIELD_40"].replace("02 05 08 11",'999')
        self.num_data["FIELD_40"] = convert_to_int(self.num_data["FIELD_40"])

        self.num_data["FIELD_40"] = self.num_data["FIELD_40"].replace(999,-1)


        ##FIELD_45
        self.not_num_data = self.num_data[['FIELD_1','FIELD_12','FIELD_14','FIELD_15','FIELD_32','FIELD_33','FIELD_34','FIELD_45','FIELD_46','FIELD_21','FIELD_40']]                     
        self.num_data = self.num_data.drop(['FIELD_1','FIELD_12','FIELD_14','FIELD_15','FIELD_32','FIELD_33','FIELD_34','FIELD_45','FIELD_46','FIELD_21','FIELD_40'],axis=1)
        self.later_processed =  self.not_num_data[['FIELD_21','FIELD_40']]
        self.not_num_data = self.not_num_data.drop(['FIELD_21','FIELD_40'],axis=1)
        
        self.not_num_data["FIELD_12"] = self.not_num_data["FIELD_12"].replace(0,False)
        self.not_num_data["FIELD_12"] = self.not_num_data["FIELD_12"].replace(1,True)
        self.not_num_data["FIELD_45"] = self.not_num_data["FIELD_45"].replace(1,False)
        self.not_num_data["FIELD_45"] = self.not_num_data["FIELD_45"].replace(2,True)
        self.not_num_data["FIELD_1"] = self.not_num_data["FIELD_1"].replace(0,False)
        self.not_num_data["FIELD_1"] = self.not_num_data["FIELD_1"].replace(1,True)
        self.not_num_data["FIELD_14"] = self.not_num_data["FIELD_14"].replace(0,False)
        self.not_num_data["FIELD_14"] = self.not_num_data["FIELD_14"].replace(1,True)
        self.not_num_data["FIELD_15"] = self.not_num_data["FIELD_15"].replace(0,False)
        self.not_num_data["FIELD_15"] = self.not_num_data["FIELD_15"].replace(1,True)
        self.not_num_data["FIELD_32"] = self.not_num_data["FIELD_32"].replace(0,False)
        self.not_num_data["FIELD_32"] = self.not_num_data["FIELD_32"].replace(1,True)
        self.not_num_data["FIELD_33"] = self.not_num_data["FIELD_33"].replace(0,False)
        self.not_num_data["FIELD_33"] = self.not_num_data["FIELD_33"].replace(1,True)
        self.not_num_data["FIELD_34"] = self.not_num_data["FIELD_34"].replace(0,False)
        self.not_num_data["FIELD_34"] = self.not_num_data["FIELD_34"].replace(1,True)
        self.not_num_data["FIELD_46"] = self.not_num_data["FIELD_46"].replace(0,False)
        self.not_num_data["FIELD_46"] = self.not_num_data["FIELD_46"].replace(1,True)
        self.boolean_data['FIELD_1'] = self.not_num_data['FIELD_1']
        self.boolean_data['FIELD_12'] = self.not_num_data['FIELD_12']
        self.boolean_data['FIELD_14'] = self.not_num_data['FIELD_14']
        self.boolean_data['FIELD_15'] = self.not_num_data['FIELD_15']
        self.boolean_data['FIELD_32'] = self.not_num_data['FIELD_32']
        self.boolean_data['FIELD_33'] = self.not_num_data['FIELD_33']
        self.boolean_data['FIELD_34'] = self.not_num_data['FIELD_34']
        self.boolean_data['FIELD_45'] = self.not_num_data['FIELD_45']
        self.boolean_data['FIELD_46'] = self.not_num_data['FIELD_46']
        
        self.boolean_data["FIELD_2"] = self.boolean_data["FIELD_2"].replace(0,False)
        self.boolean_data["FIELD_2"] = self.boolean_data["FIELD_2"].replace(1,True)

        # DATA IMPUTATION
        self.isnull_features = self.boolean_data.isnull()
        
        ## Boolean Data
        def boolean_missing_data_handler(col_array):
            for col in col_array:
                temp = []
                for value in self.boolean_data[str(col)]:
                    if value == True:
                        temp.append(1)
                    elif value == False:
                        temp.append(0)
                    elif value == 'None':
                        temp.append(-1)
                    else:
                        temp.append(-999)
                self.boolean_data[col] = temp
        boolean_missing_data_handler(['FIELD_2', 'FIELD_18', 'FIELD_19', 'FIELD_20', 'FIELD_23', 'FIELD_25','FIELD_26', 'FIELD_27', 'FIELD_28', 'FIELD_29', 'FIELD_30', 'FIELD_31','FIELD_36', 'FIELD_37', 'FIELD_38', 'FIELD_47', 'FIELD_48', 'FIELD_49','FIELD_1', 'FIELD_12', 'FIELD_14', 'FIELD_15', 'FIELD_32', 'FIELD_33','FIELD_34', 'FIELD_45', 'FIELD_46'])

        ## Numeric data
        from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler
        # Use 3 nearest rows which have a feature to fill in each row's missing features
        self.X_filled_knn = KNN(k=3).fit_transform(self.num_data)

        self.df_filled = pd.DataFrame(data=self.X_filled_knn,    # values
                #                  index=X_filled_knn[1:,0],    # 1st column as index
                                 columns=self.num_data.columns)  # 1st row as the column names
        self.df_filled = self.df_filled.round(2)

        self.num_data = self.df_filled
        
        ##Categorical Data
        for i in self.cat_data.columns:
                self.cat_data[i] = self.cat_data[i].str.lower()

        self.cat_data_isnull = self.cat_data.isnull()

        def isnull_handler(col_array):
            for col in col_array:
                temp = []
                for value in self.cat_data_isnull[str(col)]:
                    if value == True:
                        temp.append(1)
                    elif value == False:
                        temp.append(0)
                    elif value == 'None':
                        temp.append(-1)
                    else:
                        temp.append(-999)
                self.cat_data_isnull[str(col)+"_isnull"] = temp
        self.col_array = ['province', 'district', 'maCv', 'FIELD_7', 'FIELD_8', 'FIELD_9',
            'FIELD_10', 'FIELD_13', 'FIELD_17', 'FIELD_24', 'FIELD_35', 'FIELD_39',
            'FIELD_41', 'FIELD_42', 'FIELD_43', 'FIELD_44']
        isnull_handler(self.col_array)
        
        for col in self.col_array:
            self.cat_data_isnull = self.cat_data_isnull.drop([col],axis=1)

        ##FIELD 7
        import ast
        count_FIELD_7 = []
        for i in self.cat_data['FIELD_7']:
            try:
                count_FIELD_7.append(len(ast.literal_eval(i)))
            except ValueError:
                count_FIELD_7.append(0)

        temp = []
        for value in self.cat_data['FIELD_7']:
            try:
                temp.append(sorted(ast.literal_eval(value)))
            except ValueError:
                temp.append([])
        self.cat_data['FIELD_7'] = temp

        def fill_nan(col_array):
            for col in col_array:
                temp = []
                for value in self.cat_data[str(col)]:
                    try:
                        if value == 'none':
                            temp.append('missing')
                        elif isinstance(value,float) == True:
                            temp.append('missing')
                        else:
                            temp.append(value)
                    except ValueError:
                        temp.append('missing')
                if len(temp) != len(self.cat_data):
                    print(len(temp),"failed")
                    break
                self.cat_data[col] = temp
        fill_nan(['province', 'district', 'maCv', 'FIELD_7', 'FIELD_8', 'FIELD_9',
            'FIELD_10', 'FIELD_13', 'FIELD_17', 'FIELD_24', 'FIELD_35', 'FIELD_39',
            'FIELD_41','FIELD_42','FIELD_43', 'FIELD_44'])
        self.later_processed.FIELD_21[0]>0

        def fill_nan_later_processed(col_array):
            for col in col_array:
                temp = []
                for value in self.later_processed[str(col)]:
                    try:
                        if value == 'none':
                            temp.append('-999')
                        elif (isinstance(value,float) == True) and (value >0):
                            temp.append(value)
                        else:
                            temp.append('-999')
                    except ValueError:
                        temp.append('-999')
                if len(temp) != len(self.cat_data):
                    print("failed")
                    break
                self.later_processed[col] = temp
        fill_nan_later_processed(['FIELD_21', 'FIELD_40'])
        ## Count Encoding
        self.cat_data.FIELD_7 = self.cat_data.FIELD_7.astype(str)
        def count_encode(X, categorical_features, normalize=False):
            X_ = pd.DataFrame()
            for cat_feature in categorical_features:
                X_[cat_feature] = X[cat_feature].astype(
                    'object').map(X[cat_feature].value_counts())
                if normalize:
                    X_[cat_feature] = X_[cat_feature] / np.max(X_[cat_feature])
            X_ = X_.add_suffix('_count_encoded')
            if normalize:
                X_ = X_.astype(np.float32)
                X_ = X_.add_suffix('_normalized')
            else:
                try:
                    X_ = X_.astype(np.uint32)
                except:
                    print("the invalid value is: ",X_)
            return X_
        for i in self.cat_data.columns:
            if len(self.cat_data[i].unique())>10:
                self.cat_data[i] = count_encode(self.cat_data, [i])

        ## OneHot Encoding
        for i in self.cat_data.columns:
            if len(self.cat_data[i].unique())<=10:
                df_onehot = pd.get_dummies(self.cat_data[i])
                for k in range(0,df_onehot.shape[1]):
                    self.cat_data[str(df_onehot.columns[k]) + "_onehot"] = df_onehot.iloc[ : , k]
                self.cat_data = self.cat_data.drop([i],axis=1)

        # Combine All Data
        def combine_dataframe(df):
            for i in range(0,df.shape[1]):
                self.cat_data[str(df.columns[i])] = df.iloc[ : , i]
            return "succeed!"
        combine_dataframe(self.later_processed)
        self.cat_data['count_FIELD_7'] = count_FIELD_7
        combine_dataframe(self.boolean_data)
        def combine_dataframe(df):
            for i in range(0,df.shape[1]):
                self.num_data[str(df.columns[i])] = df.iloc[ : , i]
            return "succeed!"
        combine_dataframe(self.cat_data)
        self.data = self.num_data
        try:
            self.data = self.data.drop(['label','g2_onehot'],axis=1)
        except KeyError:
            pass
        columns = ['FIELD_3', 'FIELD_4', 'FIELD_5', 'FIELD_6', 'FIELD_11', 'FIELD_16', 'FIELD_22', 'FIELD_50', 'FIELD_51', 'FIELD_54', 'FIELD_55', 'age_source', 'FIELD_567', 'FIELD_523', 'province', 'maCv', 'FIELD_7', 'FIELD_9', 'FIELD_13', 'FIELD_39',  'FIELD_21', 'FIELD_40', 'count_FIELD_7', 'FIELD_2', 'FIELD_29', 'FIELD_30', 'FIELD_31', 'FIELD_36', 'FIELD_37', 'FIELD_47', 'FIELD_48', 'FIELD_49', 'FIELD_1', 'FIELD_12', 'FIELD_14', 'FIELD_15', 'FIELD_32', 'FIELD_33', 'FIELD_34', 'FIELD_45', 'FIELD_46']
        #self.data = self.data.drop(['FIELD_38','FIELD_18','FIELD_19','FIELD_20','FIELD_23','FIELD_25','FIELD_26','FIELD_27','FIELD_28','district'],axis=1)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        print("Data Shape to train: " ,self.data.shape)
        return self.data[columns], self.label


if __name__ == "__main__":
    obj_train = eda_n_cleaning("data/train.csv")
    data_train, data_train_label = obj_train.eda()
    obj_test = eda_n_cleaning("data/test.csv")
    data_test, data_test_label = obj_test.eda()
    data_test.to_csv('data_test.csv', index=False)
    data_train.to_csv('data_train.csv', index=False)
    data_train_label.to_csv('data_train_label.csv', index=False)
