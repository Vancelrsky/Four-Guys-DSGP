get_ipython().run_line_magic('run', 'MyNoteBook.py')
import Functions
from sklearn.decomposition import PCA
import pandas as pd
import gzip
import numpy as np
with gzip.open('cleaned_data.zip','rb') as data:
    data = pd.read_csv(data,index_col=[0,1])

resting = ['SLEEPING','LYING_DOWN']


phone_state = ['PHONE_IN_POCKET','PHONE_IN_HAND', 'PHONE_IN_BAG', 'PHONE_ON_TABLE']
body_state = ['SITTING','FIX_walking', 'FIX_running', 'OR_standing']
loc_state = ['OR_indoors', 'OR_outside', 'LOC_home', 'LOC_main_workplace','AT_SCHOOL','IN_A_CAR', 'ON_A_BUS']
focus = ['LAB_WORK', 'IN_CLASS', 'IN_A_MEETING','COMPUTER_WORK', 'WITH_CO-WORKERS', 'DRIVE_-_I_M_THE_DRIVER','TALKING']

housekeeping = ['COOKING', 'BATHING_-_SHOWER', 'CLEANING', 'DOING_LAUNDRY', 'WASHING_DISHES', 'EATING', 'TOILET', 'GROOMING', 'DRESSING']
exercising = ['BICYCLING','OR_exercise','STAIRS_-_GOING_UP','STAIRS_-_GOING_DOWN', 'ELEVATOR']
quiet_entertainment = ['FIX_restaurant','WATCHING_TV', 'SURFING_THE_INTERNET', 'LOC_beach','WITH_FRIENDS','DRIVE_-_I_M_A_PASSENGER']
busy_entertainment = ['SHOPPING', 'STROLLING', 'DRINKING__ALCOHOL_', 'AT_A_PARTY', 'AT_A_BAR','SINGING']
main_label_list = []

new_label_list = ['sleep','entertainment','exercise','life_activity','efficiency','on_the_way']
new_label_dict = {'sleep':0,'entertainment':1,'exercise':2,'life_activity':3,'efficiency':4,'on_the_way':5,'Normal':6}
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


    for i in range(0,s[0]): #��ÿ��timestamps
        arr = np.where(Y[i]==1) #�����@��timestamp ��Щlabel��ture
        temp = []
        for j in arr[0]:
            temp.append(label_names[j]) #���@��timestamp true��label nameƴ��list
        label_pair.loc[timestamps[i], 'Label Name'] = temp #��list���M������dataframeλ��

    new_label = []
    for index in range(len(label_pair.index)):
        label = label_pair.iloc[index].values[0]
        print(label)
        break
        for num,status in enumerate(main_label_list):
            if bool(set(status) & set(label)):
                new_label.append(num)
                break
            elif bool(set(label) & set(all_label_list)):
                continue
            else:
                new_label.append(new_label_dict['Normal'])
                break

    #muti_index = pd.MultiIndex.from_product([[uuid], X.index], names=['uuid','timestamps'])
    #new_label = pd.DataFrame(data = new_label, index = muti_index,columns = ['Status'])
    #new_label_data = pd.concat([new_label_data,new_label],axis=0,ignore_index=False)

new_label_data.value_counts()
with gzip.open('cleaned_data.zip','rb') as data:
    data = pd.read_csv(data,index_col=[0,1])
len(data.groupby('uuid').count().index)
main_label_list = [['SLEEPING'],
                   ['FIX_restaurant','SHOPPING', 'STROLLING', 'DRINKING__ALCOHOL_','WATCHING_TV', 'SURFING_THE_INTERNET', 'AT_A_PARTY', 'AT_A_BAR', 'LOC_beach', 'SINGING', 'WITH_FRIENDS'],
                   ['FIX_walking', 'FIX_running', 'BICYCLING','OR_exercise'],
                   ['COOKING', 'BATHING_-_SHOWER', 'CLEANING', 'DOING_LAUNDRY', 'WASHING_DISHES', 'EATING', 'TOILET', 'GROOMING', 'DRESSING'],
                   ['LAB_WORK', 'IN_CLASS', 'IN_A_MEETING', 'LOC_main_workplace','COMPUTER_WORK','AT_SCHOOL', 'WITH_CO-WORKERS'],
                   ['IN_A_CAR', 'ON_A_BUS', 'DRIVE_-_I_M_THE_DRIVER', 'DRIVE_-_I_M_A_PASSENGER','STAIRS_-_GOING_DOWN', 'ELEVATOR']]
new_label_list = ['sleep','entertainment','exercise','life_activity','efficiency','on_the_way']
new_label_dict = {'sleep':0,'entertainment':1,'exercise':2,'life_activity':3,'efficiency':4,'on_the_way':5,'Normal':6}
all_label_list = []

for i in main_label_list:
    all_label_list = all_label_list + i
uuid_list = list(data.groupby('uuid').count().index)
new_label_data = pd.DataFrame()
for uuid in [uuid_list[0]]:
    X,Y,M,timestamps,feature_names,label_names = Functions.read_user_data(uuid)
    label_pair = pd.DataFrame(
        columns = ['Label Name'],
        index = timestamps
    )
    s = Y.shape


    for i in range(0,s[0]): #��ÿ��timestamps
        arr = np.where(Y[i]==1) #�����@��timestamp ��Щlabel��ture
        temp = []
        for j in arr[0]:
            temp.append(label_names[j]) #���@��timestamp true��label nameƴ��list
        label_pair.loc[timestamps[i], 'Label Name'] = temp #��list���M������dataframeλ��
    new_label = []

    for index in label_pair.index:
        label = label_pair.loc[index].values[0]
        if bool(label) == True:
            for num,status in enumerate(main_label_list):
                if bool(set(status) & set(label)):
                    new_label.append(num)
                    break
                elif bool(set(label) & set(all_label_list)):
                    continue
                else:
                    new_label.append(new_label_dict['Normal'])
                    break 

    #muti_index = pd.MultiIndex.from_product([[uuid], X.index], names=['uuid','timestamps'])
    #new_label = pd.DataFrame(data = new_label, index = muti_index,columns = ['Status'])
    #new_label_data = pd.concat([new_label_data,new_label],axis=0,ignore_index=False)
X_train = pd.DataFrame(X_train,columns=data.iloc[:,:-1].columns)
X_test = pd.DataFrame(X_test,columns=data.iloc[:,:-1].columns)
def pca_to_data(csv_df,n):

    pca = PCA(n_components=n)
    features = csv_df.loc[:,csv_df.columns.str.startswith('audio_naive')]
    new_features = pca.fit_transform(features)
    pca_components = pca.components_

    print('PCA explained variance ratio is', pca.explained_variance_ratio_.sum())

    new_feature_df = pd.DataFrame(data=new_features,index=csv_df.index,columns=['audio_naive:pc1','audio_naive:pc2'])
    other_features = csv_df.loc[:,(csv_df.columns.str.startswith('audio_naive') == False)]
    new_feature_df = pd.concat([other_features,new_feature_df],axis=1,ignore_index=False)


    return (new_feature_df, pca_components)
def pca_to_train_test(X_train, X_test):
    X_train = pd.DataFrame(X_train,columns=data.iloc[:,:-1].columns)
    X_test = pd.DataFrame(X_test,columns=data.iloc[:,:-1].columns)
    X_train_pca, projection = pca_to_data(X_train,2)

    audio_test = X_test.loc[:,X_test.columns.str.startswith('audio_naive')]
    X_test_pca = X_test.loc[:,X_test.columns.str.startswith('audio_naive')==False]
    projection_matrix = pd.DataFrame(np.dot(audio_test,projection.T),columns=['audio_naive:pc1','audio_naive:pc2'])
    X_test_pca = pd.concat([X_test_pca,projection_matrix],axis=1)
    return X_train_pca.values, X_test_pca.values
from sklearn.model_selection import train_test_split
with gzip.open('cleaned_data.zip','rb') as file:
    feature_data = pd.read_csv(file,index_col=[0,1])

with gzip.open('new_label_data.zip','rb') as file:
    new_label_data = pd.read_csv(file,index_col=[0,1])

data = pd.concat([feature_data,new_label_data],join='inner',ignore_index=False,axis=1)
X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values
# split data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2, random_state = 52)
X_train, X_test = Functions.pca_to_train_test(X_train,X_test,data)

print(X_train.shape)
print(X_test.shape)
l_dict = {}
for key in label_pair.values:
    l_dict[key] = l_dict.get(key, 0) + 1
l_dict
Functions.get_related_label('COMPUTER_WORK')

each
new_label_data.value_counts()
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
with gzip.open('cleaned_data.zip','rb') as file:
    feature_data = pd.read_csv(file,index_col=[0,1])
# split data
X_train, X_test, Y_train, Y_test = train_test_split(feature_data,new_label_data,test_size= 0.2, random_state = 6)

# fit model 
model = XGBClassifier()
model.fit(X_train, Y_train)
print(model)
# make predictions for test data
Y_pred = model.predict(X_test)
predictions = [round(value) for value in Y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
