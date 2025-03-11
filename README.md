# Code Reference for Pandas
## Install Packages
```Bash
python -m pip install pandas
```
## Include Packages
```Python
import pandas as pd
```
## Series
A series is the basic one-dimensional data structure in pandas. The constainer consists of an axis label and a numpy array structure.

In general, a pandas series can be created using the corresponding constructor
```Python
class pandas.Series(data=None,		# array-like, Iterable, dict, scalar value
  		    index=None,		# array-like of Index (1-dim)
		    dtype=None,		# str, np.dtype, ExtensionDtype, optional
		    name=None,		# Hashable, default None
	            copy=None,		# bool, default False 
		    fastpath=False)
```
We now show several examples for the creation of series in pandas
```Python
s1 = pd.Series([1, 3, 5, np.nan, 6, 8])
s2 = pd.Series(data=np.array([2, 5, 8]), index=["a", "b", "c"])
```
This results in
```
s1
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```
and
```
s2
a    2
b    5
c    8
dtype: int32
```
Additionally, a series can be created from a dictionary
```Python
s3 = pd.Series({"b": 1, "a": 0, "c": 2})
```
### Attributes
```Python
Series.index			# The index (axis labels) of the Series
Series.values			# Return Series as ndarray or ndarray-like depending on the dtype
Series.dtype			# Return the dtype object of the underlying data
Series.hasnans			# Return True if there are any NaNs
Series.dtypes			# Return the dtype object of the underlying data
Series.index			# The index (axis labels) of the Series
```
### Conversion
```Python
Series.copy([deep])				# Make a copy of this object's indices and data
Series.to_numpy([dtype, copy, na_value])	# A NumPy ndarray representing the values in this Series or Index
Series.to_period([freq, copy])			# Convert Series from DatetimeIndex to PeriodIndex
Series.to_timestamp([freq, how, copy])		# Cast to DatetimeIndex of Timestamps, at beginning of period
Series.to_list()				# Return a list of the values
```
### Indexing, iteration
```Python
Series.at					# Access a single value for a row/column label pair
Series.iat					# Access a single value for a row/column pair by integer position
Series.loc					# Access a group of rows and columns by label(s) or a boolean array
Series.iloc					# Purely integer-location based indexing for selection by position
```
### Reindexing, selection, label manipulation, missing data handling
```Python
Series.head([n])					# Return the first n rows
Series.drop_duplicates(*[, keep, inplace, ...])		# Return Series with duplicate values removed
Series.duplicated([keep])				# Indicate duplicate Series values
Series.isin(values)					# Whether elements in Series are contained in values
Series.tail([n])					# Return the last n rows
Series.dropna(*[, axis, inplace, how, ...])		# Return a new Series with missing values removed
Series.ffill(*[, axis, inplace, limit, downcast])	# Fill NA/NaN values by propagating the last valid observation to next valid
Series.fillna([value, method, axis, ...])		# Fill NA/NaN values using the specified method
Series.isna()						# Detect missing values
Series.isnull()						# Series.isnull is an alias for Series.isna
Series.notna()						# Detect existing (non-missing) values
Series.notnull()					# Series.notnull is an alias for Series.notna
```
### Index and Values
The index and values of a series can be extracted as follows
```Python
pd.Series({"a": 1, "b": 2, "c": 5}).index 	# Index(['a', 'b', 'c'], dtype='object')
pd.Series([1, 2, 3]).index 			# RangeIndex(start=0, stop=3, step=1)

pd.Series({"a": 1, "b": 2, "c": 5}).values	# array([1, 2, 5], dtype=int64)
pd.Series([1, 2, 3]).values			# array([1, 2, 3], dtype=int64)
```
### Value extraction
Values of a series can be extracted as follows
```Python
s4 = pd.Series(np.array([2.0, 5.3]))
s4[1]	# 5.3

s5 = pd.Series({"a": 1, "b": -2.5})
s5[0]	# 1.0
s5["b"] # -2.5
```
#### Examples
```Python
s1 = pd.Series([1, 2, 3])
s2 = 0
```
The output of a series looks as follows
```
0    1
1    2
2    3
dtype: int64
```
## Indexes
An index is an immutable sequence used for indexing and alignment. 

The constructor of a pandas index looks as follows
```Python
class pandas.Index(data=None,		# array-like (1-dimensional)
                   dtype=None,		# NumPy dtype (default: object)
                   copy=False,		# bool
                   name=None,		# object
                   tupleize_cols=True	# bool (default: True)
                  )
```
## Data Frame
A data frame is the basic two-dimensional tabular data structure in pandas. The container consists of an index for rows and columns, respectively.

The constructor of a pandas data frame is given as follows
```Python
class pandas.DataFrame(data=None,	# ndarray (structured or homogeneous), Iterable, dict, or DataFrame
		       index=None,	# Index or array-like
		       columns=None,	# Index or array-like
		       dtype=None,	# dtype, default None
		       copy=None)	# bool or None, default None
```
As an example we consider the following data frame
```Python
df = pd.DataFrame(data={"name": ["A", "B", "C"],
                        "weight": [5, -2, 3],
                        "abs": [1.25, 9.24, -0.12]})
```
This results in 
```
	name	weight	abs
0	A	5	1.25
1	B	-2	9.24
2	C	3	-0.12
```
Furthermore, we can create data frames as follows
```Python
d = {"column_1": [1, 2], "column_2": [3, 4]}
df2 = pd.DataFrame(data=d)
```
### Attributes
```Python
DataFrame.index			# The index (row labels) of the DataFrame
DataFrame.columns		# The column labels of the DataFrame
DataFrame.dtypes		# Return the dtypes in the DataFrame
DataFrame.values		# Return a Numpy representation of the DataFrame
DataFrame.shape			# Return a tuple representing the dimensionality of the DataFrame
```
### Indexing
```Python
DataFrame.head([n])		# Return the first n rows
DataFrame.tail([n])		# Return the last n rows
DataFrame.at			# Access a single value for a row/column label pair
DataFrame.iat			# Access a single value for a row/column pair by integer position
DataFrame.loc			# Access a group of rows and columns by label(s) or a boolean array
DataFrame.iloc			# Purely integer-location based indexing for selection by position
```
### General methods
```Python
pivot(data, *, columns[, index, values])			# Return reshaped DataFrame organized by given index / column values
pivot_table(data[, values, index, columns, ...])		# Create a spreadsheet-style pivot table as a DataFrame
merge(left, right[, how, on, left_on, ...])			# Merge DataFrame or named Series objects with a database-style join
concat(objs, *[, axis, join, ignore_index, ...])		# Concatenate pandas objects along a particular axis
```
### Handle missing data
```Python
```
### Add columns
We can add columns via
```Python
df["factor"] = 2.0
df["weight_double"] = df["factor"] * df["weight"]
```
resulting in
```
    name	weight	abs	factor	weight_double
0	A	5	1.25	2.0	10.0
1	B	-2	9.24	2.0	-4.0
2	C	3	-0.12	2.0	6.0
```
We can use the assign and apply methods to create new columns by applying a function to a given column
```Python
# Loc
df.loc[:, "squared_value"] = df["value"].apply(lambda x: x*x)			# Single column
df.loc[:, "factor_abs"] = df.apply(lambda x: x["factor"]*x["abs"], axis=1)	# Multiple columns

# Assign
df_assigned_one = df.assign(new_variable="empty")
df_assigned_two = df.assign(new_column=df["old_column"].apply(my_function))
df_assigned_three = df.assign(
    weight_abs=lambda x: x["weight"]*x["abs"]
)
df_assigned_multiple = df.assign(factor_two=lambda x: x["factor"]*2,
                                 factor_four=lambda x: x["factor"]*4,
                                 factor_ten=lambda x: x["factor_two"]*5)
df_flag = df.assign(column_flag=df["old_column"].isin(arr_values))

# Numpy where
df_where = df_where.assign(new_col=np.where(df["column_valid"], df["value_valid"], "value_not_valid"))
```
We can also include lambda functions as well
```Python
df_lambda["new_column] = df_lambda["existing_column"].apply(lambda x: my_function(x) if isinstance(x, str) else "")
```
### Rename
We can rename columns as follows
```Python
df.rename(columns={"factor": "weight_factor"})
```
```
	name	weight	abs	weight_factor	weight_double
0	A	5	1.25	2.0		10.0
1	B	-2	9.24	2.0		-4.0
2	C	3	-0.12	2.0		6.0
```
### Filter
We can filter rows based on conditions as follows
```Python
df_filter = df[df["name"] == "A"].copy()
```
```
df_filter
	name	weight	abs	factor	weight_double
0	A	5	1.25	2.0	10.0
```
We can also combine different files as follows
```Python
df_filter = df[(df["name"] == "A") & (df["weight"].isin([5, -2]))]
```
In addition we can filter via loc as follows
```Python
df_filtered = df.loc[(df["name"] == "A") & (df["value"].isin([5]))]
```
We can show duplicate rows via
```Python
df_duplicates = df[df.duplicated(subset=["my_column"])]
```
We can filter for duplicate values on specific rows via the following code
```Python
df_filter_duplicates = df[~df.duplicated(subset=["my_column"])]
```
### Filling NA values
In order to fill NA values we can use the following procedure
```Python
df["filled_na"] = df["col_with_na"].fillna("to be done")
```
### Joining Data Frames
Data Frames can be joined with "merge". The syntax is as follows
```Python
DataFrame.merge(right,
                how='inner',
                on=None,
                left_on=None,
 		right_on=None,
		left_index=False,
		right_index=False,
		sort=False,
		suffixes=('_x', '_y'),
		copy=None,
		indicator=False,
		validate=None)
```
Example for a merge
```
df_1 = pd.DataFrame({"id": [1, 2, 3],
                     "value": ["a", "b", "c"]})
df_2 = pd.DataFrame({"id": [2, 3, 4],
                     "value_2": ["d", "e", "f"]})
df_3 = df_1.merge(df_2, how="inner", on="id")
```
This results in
```
df_1
 	id	value_1
0	1	a
1	2	b
2	3	c

df_2
        id	value_2
0	2	d
1	3	e
2	4	f

df_3
	id	value_1	value_2
0	2	b	d
1	3	c	e
```
## Input and Output
### Pickling
```Python
pd.read_pickle(filepath_or_buffer[, ...])               # Load pickled pandas object
DataFrame.to_pickle(path[, compression, ...])           # Pickle (serialize) object to file
```
### Flat file
```Python
# Table
pd.read_table(filepath_or_buffer, *[, sep, ...])        # Read a table data format

# CSV
pd.read_csv(filepath_or_buffer, *[, sep, ...])          # Read CSV data
DataFrame.to_csv([path_or_buf, sep, na_rep, ...])       # Write CSV data
```
### JSON
```Python
pd.read_json(path_or_buf, *[, orient, typ, ...])        # Convert a JSON string to pandas object
DataFrame.to_json([path_or_buf, orient, ...])           # Convert the object to a JSON string
```
### Excel
```Python
pd.read_excel(io[, sheet_name, header, names, ...])     # Read an Excel file into a DataFrame
DataFrame.to_excel(excel_writer[, ...])                 # Write object to an Excel sheet
```
### HTML
```Python
pd.read_html(io, *[, match, flavor, header, ...])       # Read an Excel file into a DataFrame
DataFrame.to_html([buf, columns, col_space, ...])       # Write object to an Excel sheet
```
### XML
### HDFStore
### Parquet
```Python
pd.read_parquet(path[, engine, columns, ...])           # Load a parquet object from the file path, returning a DataFrame
DataFrame.to_parquet([path, engine, ...])               # Write a DataFrame to the binary parquet format
```
### SQL
### Examples
Examples for reading files
```Python
# Read CSV file
df_csv = pd.read_csv("tmp.csv")

# Read Excel file
df_excel = pd.read_excel("tmp.xlsx", sheet_name="FirstSheet")
df_excel = pd.read_excel("tmp.xlsx", index_col=0, dtype={"Name": str, "Value": float})
df_excel = pd.read_excel(open("tmp.xlsx", "rb"), sheet_name="SecondSheet")
```
Examples for writing files
```Python
# Write output
df_out = pd.DataFrame([['a', 'b'], ['c', 'd']],
                      index=['row 1', 'row 2'],
                      columns=['col 1', 'col 2'])
# Write CSV
df_out.to_csv(index=False)

# Write Excel
df_out.to_excel("output.xlsx", sheet_name="Sheet_Name_1")  
```
Creating directories for the output
```Python
# Pathlib
from pathlib import Path
filepath = Path("folder/subfolder/output.csv")
filepath.parent.mkdir(parents=True, exists_ok=True)
df_out.to_csv(filepath)

# OS
import os
os.makedirs("folder/subfolder", exists_ok=True)
df_out.to_csv("folder/subfolder/output.csv")
```
## General functions
### Data manipulation
We list general functions for data manipulation in the following
```Python
melt(frame[, id_vars, value_vars, var_name, ...])			# Unpivot a DataFrame from wide to long format, optionally leaving identifiers set
pivot(data, *, columns[, index, values])				    # Return reshaped DataFrame organized by given index / column values
merge(left, right[, how, on, left_on, ...])				    # Merge DataFrame or named Series objects with a database-style join
concat(objs, *[, axis, join, ignore_index, ...])			# Concatenate pandas objects along a particular axis
```
### Concatenate Data Frames
We can concatenate two (or multiple) data frames by using the concat method specifying the relevant axis for concatenation.

#### Append rows
```Python
df_concat = pd.concat([df_1, df_2])
df_concat_rows = pd.concat([df_1, df_2], axis=0)
```
#### Append columns
```Python
df_concat_cols = pd.concat([df_1, df_2], axis=1)
```
### Timelike data handling
```Python
to_datetime(arg[, errors, dayfirst, ...])				# Convert argument to datetime
to_timedelta(arg[, unit, errors])					# A
date_range([start, end, periods, freq, tz, ...])			# A
```
We can specify the following frequencies
|Alias|Description|
|-|-|
|YS|Year start frequency|
|YE|Year end frequency|
|M|Monthly frequency|
|MS|Month start frequency|
|ME|Month end frequency|
|W|Weekly frequency|
|D|Calendar day frequency|
|B|Business day frequency|
|h|Hourly frequency|
|min|Minutely frequency|
|s|Secondly frequency|
|ms|Millisecond frequency|

We can also specify a given number of granularity for the frequency argument, e.g.
|Alias|Description|
|-|-|
|2D|2 calendar days frequency|
|30min|30 minutes frequency|
|30s|30 seconds frequency|

Examples
```Python
pd.to_datetime(['2018-10-26 12:00 -0500', '2018-10-26 13:00 -0500'])
# DatetimeIndex(['2018-10-26 12:00:00-05:00', '2018-10-26 13:00:00-05:00'],
#               dtype='datetime64[ns, UTC-05:00]', freq=None)

pd.date_range(start='1/1/2018', end='1/08/2018')
# DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
#                '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
#               dtype='datetime64[ns]', freq='D')

pd.date_range(start='1/1/2018', periods=8)
# DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
#                '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
#               dtype='datetime64[ns]', freq='D')

pd.date_range(start="2023-01-01", end="2023-12-31")
# DatetimeIndex(['2023-01-01', '2023-01-02', ..., '2023-12-31'],
                dtype='datetime64[ns]', length=365, freq='D')
```
## References
The pandas documentation can be found on: [https://pandas.pydata.org/docs/index.html](https://pandas.pydata.org/docs/index.html)
