import os 
os.system('Python MyNoteBook.py')
import Valid_Datasets

uuid_list = []
f = open('UUID List.txt', 'r')
for line in f.readlines():
    uuid_list.append(line.strip())
for uuid in uuid_list:
    data = Valid_Datasets.get_df(uuid)
    try:
        cleaned_data = Valid_Datasets.non_watch_value_imputer(data)
        cleaned_data = Valid_Datasets.KNN_for_watch_data(cleaned_data,10)
        cleaned_data.to_csv('Cleaned_data/%s.csv'%uuid,mode = 'w')
    except:
        print(uuid, 'failed to output')