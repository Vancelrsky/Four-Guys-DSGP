import os 
os.system('Python MyNoteBook.py')
import Valid_Datasets
import pandas as pd

""" 
uuid_list = []
f = open('UUID List.txt', 'r')
for line in f.readlines():
    uuid_list.append(line.strip())
for uuid in uuid_list:
    data = Valid_Datasets.get_df(uuid)
    try:
        # fill in with mean values
        cleaned_data = Valid_Datasets.non_watch_value_imputer(data)
        # fill watch data by using KNN
        cleaned_data = Valid_Datasets.KNN_for_watch_data(cleaned_data,10)
        # use PCA to get lower dimension for Audio data
        cleaned_data = Valid_Datasets.pca_to_data(cleaned_data,2)
        cleaned_data.to_csv('Cleaned_data/%s.csv'%uuid,mode = 'w')
    except:
        print(uuid, 'failed to output') 
"""

uuid_list = Valid_Datasets.get_cross_validation('train',0)
valid_data = pd.DataFrame()
for uuid in uuid_list:
    df = Valid_Datasets.get_df(uuid)
    try:
        cleaned_data = Valid_Datasets.non_watch_value_imputer(df)
        cleaned_data = Valid_Datasets.KNN_for_watch_data(cleaned_data,10)
        muti_index = pd.MultiIndex.from_product([[uuid], cleaned_data.index], names=['uuid','timestamps'])
        cleaned_data = pd.DataFrame(cleaned_data.values, columns=cleaned_data.columns, index=muti_index)
        valid_data = pd.concat([valid_data,cleaned_data],axis=0,ignore_index=False)
    except:
        print(uuid,'failed to export')


valid = Valid_Datasets.pca_to_data(valid_data,2)
valid.to_csv('train_data.csv',mode = 'w')