# this is for my exercises
# here I intend to put all the useful re-usable fuctions in this file (package/module)
import numpy as np
import pandas as pd

# the input should be a data frame
def handle_non_numeric_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
            
        if df[column].dtype != np.int64 or df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elems = set(column_contents)
            
            idx = 0
            for elem in unique_elems:
                if elem not in text_digit_vals:
                    text_digit_vals[elem] = idx
                    idx+=1
            
            df[column] = list(map(convert_to_int, df[column]))
            
    return df