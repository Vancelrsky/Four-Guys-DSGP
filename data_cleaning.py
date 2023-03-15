get_ipython().run_line_magic('run', 'MyNoteBook.py')
import Functions
from sklearn.decomposition import PCA
import pandas as pd
import gzip
uuid_list = []
f = open('UUID List.txt', 'r')
for line in f.readlines():
    uuid_list.append(line.strip())
valid_data = pd.DataFrame()
for uuid in uuid_list:
    df = Functions.get_df(uuid)
    try:
        cleaned_data = Functions.non_watch_value_imputer(df)
        cleaned_data = Functions.KNN_for_watch_data(cleaned_data,10)
        muti_index = pd.MultiIndex.from_product([[uuid], cleaned_data.index], names=['uuid','timestamps'])
        cleaned_data = pd.DataFrame(cleaned_data.values, columns=cleaned_data.columns, index=muti_index)
        valid_data = pd.concat([valid_data,cleaned_data],axis=0,ignore_index=False)
    except:
        print(uuid,'failed to export')
valid_data.to_csv('cleaned_data.zip',mode = 'w',compression= 'gzip')
new_label_data = pd.DataFrame()
with gzip.open('cleaned_data.zip','rb') as data:
    data = pd.read_csv(data,index_col=[0,1])
for uuid in data.groupby('uuid').count().index:
    X,Y,M,timestamps,feature_names,label_names = Functions.read_user_data(uuid)
    label = pd.DataFrame(data=Y,columns=label_names)
    main_label_list = ['LYING_DOWN','SITTING','FIX_walking','FIX_running','BICYCLING','OR_standing']
    label = label[main_label_list]
    new_label = []
    for i in label.index:
        if label.loc[i,:].values.any() == False:
            new_label.append('Other')
        else:
            for j in main_label_list:
                if label.loc[i,j] == True:
                    new_label.append(j)
    muti_index = pd.MultiIndex.from_product([[uuid], X.index], names=['uuid','timestamps'])
    new_label = pd.DataFrame(data = new_label, index = muti_index,columns = ['Status'])
    new_label_data = pd.concat([new_label_data,new_label],axis=0,ignore_index=False)

new_label_data.to_csv('new_label_data.zip',mode = 'w',compression= 'gzip')
