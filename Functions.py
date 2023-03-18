import numpy as np
import gzip
import pandas as pd
from sklearn.impute import KNNImputer 
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
# this method take a dataframe as input, return the feature part and label part
def parse_header_of_csv(csv_df):
    # Isolate the headline columns:

    for (ci,col) in enumerate(csv_df.columns):
        # find the start of label column
            if col.startswith('label:'):
                first_label_ind = ci
                break
            pass
    # use the "start of label" find above to split feature and label
    feature_names = csv_df.columns[1:first_label_ind]
    label_names = list(csv_df.columns[first_label_ind:-1])

    # remove "label: " get pure label name
    for (li,label) in enumerate(label_names):
    # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
            assert label.startswith('label:')
            label_names[li] = label.replace('label:','')
            pass

    csv_df.rename(columns=dict(zip(csv_df.columns[first_label_ind:-1],label_names)),inplace=True)
        
    return (feature_names,label_names)

"""
this method take a dataframe and number of features as input, 
return sensor matrix, label matrix, missing label matrix and timestamp matrix(index)
"""
def parse_body_of_csv(csv_df,n_features):


    # Read the entire CSV body into a single numeric matrix:
    
    # Timestamp is the primary key for the records (examples):
    timestamps = csv_df.index
    # Read the sensor features:
    X = csv_df[csv_df.columns[0:n_features+1]]
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = csv_df[csv_df.columns[n_features+1:-1]] # This should have values of either 0., 1. or NaN

    M = pd.isna(trinary_labels_mat) # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0. # Y is the label matrix

    
    return (X,Y,M,timestamps)

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
this method take id of subject as input
return sensor matrix, label matrix, missing label matrix and timestamp matrix(index) by calling parse_body_of_csv()method

'''
def read_user_data(uuid):
    user_data_file = 'Datasets/%s.features_labels.csv.gz' % uuid

    with gzip.open(user_data_file,'rb') as fid:
        csv_df = pd.read_csv(fid,delimiter=',', index_col= 0)
        pass

    (feature_names,label_names) = parse_header_of_csv(csv_df)
    n_features = len(feature_names)
    (X,Y,M,timestamps) = parse_body_of_csv(csv_df,n_features)

    return (X,Y,M,timestamps,feature_names,label_names)

#To create uuid_list which includes all uuid
uuid_list = []
f = open('UUID List.txt', 'r')
for line in f.readlines():
    uuid_list.append(line.strip())

# To create main feature list
    main_feature = []
    f = open('Main Feature.txt', 'r')
    for line in f.readlines():
        main_feature.append(line.strip())

"""
by calling this method we can get a list of dataframe which contain all the user's sensor data
//3.6 v0 may get label lists later w.
Author chen
"""
def get_df_list():
    #To create uuid_list which includes all uuid
    uuid_list = []
    f = open('UUID List.txt', 'r')
    for line in f.readlines():
        uuid_list.append(line.strip())

    main_feature = []
    f = open('Main Feature.txt', 'r')
    for line in f.readlines():
        main_feature.append(line.strip())

    instance = []
    # Run all uuid
    for i in range(len(uuid_list)):    
        (X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid_list[i])

        # Create dataframe for all Main Feature value
        Main_X = pd.DataFrame(X.loc[:,X.columns.str.startswith(main_feature[0])], columns = [main_feature[0]])
        for j in range(1,len(main_feature)):
            Main_X = pd.concat([Main_X, X.loc[:,X.columns.str.startswith(main_feature[j])]], axis=1)
        instance.append(Main_X)
    return instance
def get_df(uuid):
    main_feature = []
    f = open('Main Feature.txt', 'r')
    for line in f.readlines():
        main_feature.append(line.strip())

    # Run all uuid 
    (X,Y,M,timestamps,feature_names,label_names) = read_user_data(uuid)

    # Create dataframe for all Main Feature value
    Main_X = pd.DataFrame(X.loc[:,X.columns.str.startswith(main_feature[0])], columns = [main_feature[0]])
    for j in range(1,len(main_feature)):
        Main_X = pd.concat([Main_X, X.loc[:,X.columns.str.startswith(main_feature[j])]], axis=1)
    return Main_X
# 用均值补除了手表的数据
def non_watch_value_imputer(df):
    # get the data except watch
    non_watch_values = df.loc[:,(df.columns.str.startswith('watch_') == False)]
    valid_data = pd.DataFrame(columns = ['blank'])
    # use mean values to fill the none value
    for column in non_watch_values.columns:
        column_df = non_watch_values[column]
        mean_value = non_watch_values[column].mean()
        column_df = column_df.fillna(mean_value)
        valid_data = pd.concat([valid_data, column_df],axis=1,ignore_index=False)

    valid_data = valid_data[valid_data.columns[1:]]
    #combine the watch data
    combine_data = pd.concat([valid_data,df.loc[:,df.columns.str.startswith('watch_')]],axis=1,ignore_index=False)
    return combine_data
# 用其他传感器数据的KNN补手表数据
def KNN_for_watch_data(df,K):
    #input data and K neighbors
    imputer = KNNImputer(n_neighbors=K)
    df[list(df.columns)] = imputer.fit_transform(df)
    return df 
def get_cross_validation(type, folds_num):
    # 输入type与folds_num, 返回uuid_list
    uuid = []
    for fold in os.listdir('Splitted_folds'):
        if  folds_num == int(fold.split('_')[1]) and str(type).lower() == fold.split('_')[2].lower():
            uuid_list = open("Splitted_folds/%s" % fold, 'r')
            fold_uuid_list = uuid_list.read().split()
            uuid = uuid + fold_uuid_list
    return uuid
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
def pca_to_train_test(X_train, X_test, data):
    
    X_train = pd.DataFrame(X_train,columns=data.iloc[:,:-1].columns)
    X_test = pd.DataFrame(X_test,columns=data.iloc[:,:-1].columns)
    X_train_pca, projection = pca_to_data(X_train,2)

    audio_test = X_test.loc[:,X_test.columns.str.startswith('audio_naive')]
    X_test_pca = X_test.loc[:,X_test.columns.str.startswith('audio_naive')==False]
    projection_matrix = pd.DataFrame(np.dot(audio_test,projection.T),columns=['audio_naive:pc1','audio_naive:pc2'])
    X_test_pca = pd.concat([X_test_pca,projection_matrix],axis=1)
    return X_train_pca.values, X_test_pca.values
def get_related_label(char):
    with gzip.open('cleaned_data.zip','rb') as data:
        data = pd.read_csv(data,index_col=[0,1])
    new_label_data = []
    for uuid in data.groupby('uuid').count().index:
        X,Y,M,timestamps,feature_names,label_names = read_user_data(uuid)
        label_dict = {v: k for k, v in dict(enumerate(label_names + ['None'])).items()}
        label_list = []
        for each in Y:
            if np.array(each).any()==False:
                continue
            else:
                new_label_names = np.array(label_names)[each]
                if char in new_label_names:
                    label_list.append(list(new_label_names))
    new_label_data = new_label_data + label_list
    labels = []
    for i in new_label_data:
        labels = labels + i
    l_dict = {}
    for key in labels:
        l_dict[key] = l_dict.get(key, 0) + 1
    return l_dict

def splitdata(Final_data,test_size):
    with gzip.open('cleaned_data.zip','rb') as data:
        data = pd.read_csv(data,index_col=[0,1])
    idlist = data.groupby('uuid').count().index
    stamps_index = {}
    for id in idlist:
        length = len(Final_data.loc[id])
        random.seed(777)
        stamps_index[id] = random.sample(range(0,length),int(length*test_size))

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for k, v in stamps_index.items():
        test_data = pd.concat([test_data, Final_data.loc[k].iloc[v]], axis = 0)
        
        total = Final_data.loc[k]
        remain = total[~total.isin(Final_data.loc[k].iloc[v])].dropna()
        
        train_data = pd.concat([train_data, remain], axis = 0)
    
    # Prepare training data X and Y
    train_data_x = train_data.iloc[:,:-1].values
    train_data_y = train_data.iloc[:,-1].values.astype(int)

    # Prepare testing data X and Y
    test_data_x = test_data.iloc[:,:-1].values
    test_data_y = test_data.iloc[:,-1].values.astype(int)

    return train_data_x, test_data_x, train_data_y, test_data_y
