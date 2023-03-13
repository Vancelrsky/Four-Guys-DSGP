import MyNoteBook
import Valid_Datasets
instance = Valid_Datasets.get_df_list()

test_example = instance[0]

cleaned_data = Valid_Datasets.non_watch_value_imputer(test_example)
cleaned_data = Valid_Datasets.KNN_for_watch_data(cleaned_data,10)
cleaned_data
print(Valid_Datasets.get_cross_validation('test',0))
