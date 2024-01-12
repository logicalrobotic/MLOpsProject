import unittest
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import power_transform
sys.path.append('./data')
from clean_data import remove_grenades, encode_targets, encode_inputs, yeo_johnson

class TestDataCleaning(unittest.TestCase):
    def test_remove_grenades(self):
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'grenade1': [7, 8, 9], 'grenade2': [10, 11, 12]})
        expected_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        cleaned_df = remove_grenades(df)
        self.assertTrue(expected_df.equals(cleaned_df))

    def test_encoder_target(self):
        y = pd.Series(['a', 'b', 'c', 'a', 'b', 'c'])
        expected_y = np.array([0, 1, 2, 0, 1, 2])
        y_encoded = encode_targets(y)
        self.assertTrue(np.array_equal(expected_y, y_encoded))

    def test_encoder_inputs(self):
        df = pd.DataFrame({'Category': ['A', 'B', 'A', 'C', 'B'],
                'Value': [10, 20, 15, 25, 30]})
        object_cols = ['Category']
        df_encoded = encode_inputs(df, object_cols)
        # Check if the result is a DataFrame
        self.assertIsInstance(df_encoded, pd.DataFrame, "Result is not a DataFrame")
        # Check if the index is preserved
        self.assertTrue(df.index.equals(df_encoded.index), "Index mismatch")
        # Check if the column names are set correctly
        expected_column_names = ['Category_A', 'Category_B', 'Category_C']
        self.assertListEqual(list(df_encoded.columns), expected_column_names, "Incorrect column names")

    #def test_yeo_johnson(self):
        # Test case 1: Positive values
        #input_series_1 = [1, 2, 3, 4, 5]
        #transformed_arr_1 = yeo_johnson(input_series_1)
        #self.assertTrue(np.all(transformed_arr_1 >= 0), "Yeo-Johnson transformation should result in non-negative values")

        # Test case 2: Mixed positive and negative values
        #input_series_2 = [-2, -1, 0, 1, 2]
        #transformed_arr_2 = yeo_johnson(input_series_2)
        #self.assertTrue(np.all(np.isfinite(transformed_arr_2)), "Yeo-Johnson transformation should result in finite values")

        # Test case 3: Zero values
        #input_series_3 = [0, 0, 0, 0, 0]
        #transformed_arr_3 = yeo_johnson(input_series_3)
        #self.assertTrue(np.all(transformed_arr_3 == 0), "Yeo-Johnson transformation of zeros should result in zeros")

if __name__ == '__main__':
    unittest.main()
    

