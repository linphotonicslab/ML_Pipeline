# PIPELINE GUIDE
[Link to lab website](https://sites.google.com/uw.edu/photonics-lab/research?authuser=0)\
PI: Dr. Lih Lin\
Authors: [Nick Roberts](mailto:nickrob320@gmail.com) and [Dylan Jones](mailto:ddj123@uw.edu)

### Overview
The dataset used for our machine learning analysis is created using the `create_data.py` script that can be run as `python scripts/create_data.py` in your terminal.

It takes a CSV file from the [Perovskite Database](https://www.perovskitedatabase.com/Download) and converts it to a cleaned csv file ready for machine learning training.

The pipeline is **modular** and can be used as a template for different prediction targets with variable feature sets. It is built to handle the dataformatting outlined in the [Perovskite Database's PDF description of content](https://www.perovskitedatabase.com/return_databaseInstructions)

The file is structured in the following order:
1. Specify desired columns
2. Specify target column
3. Indicate the column data categories and special requirements of any columns
4. Helper functions for pipeline
5. Main data pipeline

Below is a detailed description of each section:

### Initialization
Install dependencies

```bash
# clone project
git clone https://github.com/linphotonicslab/ML_Pipeline
cd ML_Pipeline

# [OPTIONAL] create conda environment
conda create -n **ENV NAME**
conda activate **ENV NAME**

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### 1. Specify Desired Columns
The desired columns are stored in a `DESIRED_COLUMNS` dictionary, with descriptive `strings` as keys and columns `lists` that contain column names as values. How the `strings` are named is _important_ as it determines how each group of columns is formatted in the final dataframe.

In the Perovskite Database, data for multiple layers of a section of the device (e.g. A site composition) is indicate by adding a "|" character between values (Cs | FA). If the term `LAYERED` is prepended to a key string for a group of columns, it will create columns for each layer found in the datapoints labeled with `Li` where `i` is the layer number. (e.g. a_Cs_L0).

Oftentimes, one column will contain information for a particular material, and another column will have its associated numerical value (e.g. A site chemical and its associated stoichiometric ratio Cs <-> 0.23). By prepending the term `PAIRED` to a key string for a group of columns, all the columns contained will be converted into paired columns, where the column's header is the `string` for the material, and the column's value is the `float` from the numeric column. **warning: always put the `string` column first in the column lists**.

### 2. Specify Target Column
Indicate which column in `DESIRED_COLUMNS` is the target for machine learning. It is
handled quite differently in the pipeline.

### 3. Indicate the column data categories and special requirements of any columns
There are two main categories of data for the pipeline that subdivide into subcategories:
1. Non-categorical:
    - Numeric values (e.g cell area)
2. Categorical:
    - Boolean (e.g. deposition quenching)
    - String (e.g. ETL transport material)

In order to represent the categorical columns numerically for a model, the pipeline encodes there presense with one-hot encoding. To indicate whether a column should be one-hot encoded, add it to the `CATEGORICAL_COLUMNS` set.

All datapoints must be extracted from token strings formatted such that multiple materials per layer are separated by `;` and layers are separated by `|` (e.g. Cs; MA | FA) <-> (0.2, 0.8 | 1)
The input data for the `string` columns are prone to many input errors. Moreover, how `nan` inputs are handled can vary feature-to-feature depending on the usecase. Some numerical data for a particular layer ought to make physical sense (such as mixing ratios adding up to 1). The `TOKEN_DATA_FORMATTING` dictionary specifies dictionaries of key, value pairs for the function in the pipeline that handles token data extraction. The key indicates the columns where the key, value pairs are applied. Finally, `TOKEN_CLEANER_EQUIVALENTS` is a dictionary that stores equivalent chemical forms. It is used to normalize tokens to a standard chemical form (e.g. Ca -> Cs). It can also be used to convert random input errors to `nan` values.

### 4. Helper functions for the pipeline
Additional information about each function can be found in the `create_data.py` script

`modification_wrapper`:
This function wraps any function that takes in a Pandas Dataframe and returns solely a Pandas Dataframe. It prints to the console information about how the wrapped function modifys the dataframe (useful for identifying where you are losing data)

`drop_nan_in_target`:
Drops all the rows with NaN in the target column

`drop_non_abx3`:
For our study, we are not concerned with non-3D perovskites, so this function drops all those which aren't

`normalize_data_token`:
Takes an individual data token (e.g. number, chemical) and formats it by removing redudant items (e.g. parenthesis), or by converting it to a standardized form if it is listed in the `TOKEN_CLEANER_EQUIVALENTS` dictionary. It converts all strings to numeric types if possible.

`split_tokens_by_layer`:
Takes an entire datapoint and splits it into indvidual normalized tokens separated by layer.

`check_dimensions`:
Takes two lists created by `split_tokens_by_layer` and checks to see if their dimensions match. It is used if there is one set of data (headers) paired with datapoints (values).

`split`:
Wraps `split_tokens_by_layer` so that if there are key, value pairs that need to be passed into `split_tokens_by_layer` as listed in `TOKEN_DATA_FORMATTING` for specific columns, they are passed in automatically. This allows customization of `split_tokens_by_layer` for certain features.

`extract_paired_information`:
Takes information about a row of the original dataframe in the form of a pandas series and extracts all information for paired columns listed in pairs in `PAIRED_COLUMNS` and adds their information to the `row_info` dictionary that maps headers to datapoints for a given row. Also returns whether or not the data is worth including (formatted correctly)

`extract_information`:
Populates a given `row_info` dictionary with datapoints for all the non-paired columns. Also returns whether or not the data is worth including (formatted correctly)


`create_index_dictionary`:
The output dataframe is constructed from a dictionary whose keys correspond to the indices of the original dataframe that's being cleaned. For every row that passes the quality checks in `extract_information` and `extract_paired_information`, its row information stored in the `row_info` dictionary is added to the index dictionary. This allows for the easy construction of a Pandas dataframe. In a later step. This function also keeps track of all the unique columns seen.

`construct_dataframe`:
Given a set of columns and an index dictionary {row_i: {'header_i': 'value_i . . . .}}, it constructs a Pandas dataframe.

`categorical_to_one_hot`:
Converts all categorical columns to one_hot encodings of themselves. It modifies a given dataframe in place. The categorical column groups are listed in `CATEGORICAL_COLUMNS`. All these columns are stored in anon_standard_cols variable for `remove_more_than_3std_away`

`remove_more_than_3std_away`:
Removes all rows that contain values that are more than 3 standard deviation away from that values mean. It ignores categorical columns and the target column.

`remove_duplicates`:
Removes all duplicate rows (ignoring the target column) in a dataframe. For each group of duplicates, the mean or mode of the TARGET column is append to "represent" the group. Whichever is larger is chosen. This can be modified, but we found that this choice tended to work well for groups with large and small amounts of samples.

`standard_scale_features`:
Performs standard scaling on the features of the output dataframe except for the target column.

`summarize_final_df`:
Prints information about the final dataframe to the console.

### Main Pipeline

The pipeline is as follows:
1. Get only the desired columns chosen in `DESIRED_COLUMNS`
2. Drop all NaN in the target
3. Remove all non-ABX3 perovskites
4. Construct an index dictionary (information about each row in original csv stored with its index as the key)
5. Construct the output dataframe
6. Convert categorical columns to one-hot
7. Remove all rows with values more than 3std away from the value's respective mean (ignoring categorical and target columns)
8. Remove all duplicate rows
9. Standard scale features
10. Save final dataframe
