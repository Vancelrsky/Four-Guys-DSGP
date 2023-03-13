import MyNoteBook
import Valid_Datasets
instance = Valid_Datasets.get_df('00EABED2-271D-49D8-B599-1D4A09240601')

test_example = instance

cleaned_data = Valid_Datasets.non_watch_value_imputer(test_example)
cleaned_data = Valid_Datasets.KNN_for_watch_data(cleaned_data,10)
cleaned_data = Valid_Datasets.pca_to_data(cleaned_data,2)
cleaned_data
print(Valid_Datasets.get_cross_validation('test',0))
