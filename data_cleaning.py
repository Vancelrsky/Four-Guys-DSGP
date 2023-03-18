get_ipython().run_line_magic('run', 'MyNoteBook.py')
import Functions
from sklearn.decomposition import PCA
import pandas as pd
import gzip
import numpy as np
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
with gzip.open('cleaned_data.zip','rb') as data:
    data = pd.read_csv(data,index_col=[0,1])
main_label_list = [['SLEEPING'],
                   ['LAB_WORK', 'IN_CLASS', 'IN_A_MEETING', 'LOC_main_workplace','COMPUTER_WORK','AT_SCHOOL', 'WITH_CO-WORKERS'],
                   ['FIX_walking', 'FIX_running', 'BICYCLING','OR_exercise'],
                   ['COOKING', 'BATHING_-_SHOWER', 'CLEANING', 'DOING_LAUNDRY', 'WASHING_DISHES', 'EATING', 'TOILET', 'GROOMING', 'DRESSING'],
                   ['FIX_restaurant','SHOPPING', 'STROLLING', 'DRINKING__ALCOHOL_','WATCHING_TV', 'SURFING_THE_INTERNET', 'AT_A_PARTY', 'AT_A_BAR', 'LOC_beach', 'SINGING', 'WITH_FRIENDS'],                   
                   ['IN_A_CAR', 'ON_A_BUS', 'DRIVE_-_I_M_THE_DRIVER', 'DRIVE_-_I_M_A_PASSENGER','STAIRS_-_GOING_DOWN', 'ELEVATOR']]

new_label_list = ['sleep','efficiency','exercise','life_activity','entertainment','on_the_way']
new_label_dict = {'sleep':0, 'efficiency':1, 'exercise':2, 'life_activity':3, 'entertainment':4, 'on_the_way':5, 'Normal':6}
all_label_list = []

for i in main_label_list:
    all_label_list = all_label_list + i

new_label_data = pd.DataFrame()
for uuid in data.groupby('uuid').count().index:
    X,Y,M,timestamps,feature_names,label_names = Functions.read_user_data(uuid)
    label_pair = pd.DataFrame(
        columns = ['Label Name'],
        index = timestamps
    )
    s = Y.shape


    for i in range(0,s[0]): #跑每個timestamps
        arr = np.where(Y[i]==1) #尋找這個timestamp 哪些label是ture
        temp = []
        for j in arr[0]:
            temp.append(label_names[j]) #將這個timestamp true的label name拼成list
        label_pair.loc[timestamps[i], 'Label Name'] = temp #把list放進對應的dataframe位置

    new_label = []
    new_index = []
    for index in label_pair.index:
        label = label_pair.loc[index].values[0]
        if bool(label) == True:
            for num,status in enumerate(main_label_list):
                if bool(set(status) & set(label)):
                    new_label.append(num)
                    new_index.append(index)
                    break
                elif bool(set(label) & set(all_label_list)):
                    continue
                else:
                    new_label.append(new_label_dict['Normal'])
                    new_index.append(index)
                    break 

    muti_index = pd.MultiIndex.from_product([[uuid], new_index], names=['uuid','timestamps'])
    new_label = pd.DataFrame(data = new_label, index = muti_index,columns = ['Status'])
    new_label_data = pd.concat([new_label_data,new_label],axis=0,ignore_index=False)

new_label_data.to_csv('new_label_data.zip',mode = 'w',compression= 'gzip')
