get_ipython().run_line_magic('run', 'MyNoteBook.py')
import Valid_Datasets
from sklearn.decomposition import PCA
import pandas as pd
instance = Valid_Datasets.get_df('00EABED2-271D-49D8-B599-1D4A09240601')

test_example = instance

cleaned_data = Valid_Datasets.non_watch_value_imputer(test_example)
cleaned_data = Valid_Datasets.KNN_for_watch_data(cleaned_data,10)
cleaned_data = Valid_Datasets.pca_to_data(cleaned_data,2)
cleaned_data
print(Valid_Datasets.get_cross_validation('test',0))
import gzip
import pandas as pd

uuid = '00EABED2-271D-49D8-B599-1D4A09240601'
user_data_file = 'Datasets/%s.features_labels.csv.gz' % uuid
with gzip.open(user_data_file,'rb') as fid:
    csv_df = pd.read_csv(fid,delimiter=',', index_col= 0)
    csv_df = pd.DataFrame(csv_df,index=pd.MultiIndex.from_product([[uuid],csv_df.index]))
    pass
csv_df
