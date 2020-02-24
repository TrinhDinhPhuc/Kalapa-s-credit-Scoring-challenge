import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

#set to unlimited colums display
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

df = pd.read_csv("train.csv")
print(df.shape)

object_dataframe = pd.DataFrame()
bool_dataframe = pd.DataFrame()
numerical_dataframe = pd.DataFrame()
dict_types = {"object":[],"bool":[],"numerical":[]}
for (colName,colData) in df.dtypes.iteritems():
    if str(colData) == "object":
        dict_types["object"].append(colName)
    elif str(colData) == "bool":
        dict_types["bool"].append(colName)
    else:
        dict_types["numerical"].append(colName)
    

for key, value in dict_types.items():
    if key == "object":
        for i in value:
            object_dataframe[str(i)] = df[str(i)]
    elif key == "bool":
        for i in value:
            bool_dataframe[str(i)] = df[str(i)]
    else:
        for i in value:
            numerical_dataframe[str(i)] = df[str(i)]


NO_missing_values = df["province"].isnull().sum(axis=0)
NO_total_values = df["province"].count()

print(" Null: {0} \n Not Null: {1} \n total {2} \n percentage of missing values {3}".format(NO_missing_values,NO_total_values,NO_total_values + NO_missing_values,NO_missing_values/NO_total_values*100))


#print(numerical_dataframe.columns)
#print(object_dataframe["district"].value_counts().shape)
#print(object_dataframe["district"].value_counts())

plt.figure(figsize=(15,8))
sns.distplot(df["age_source2"], bins=30)
#plt.show()

