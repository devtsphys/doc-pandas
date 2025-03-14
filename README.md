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

## Timelike data handling

### Pandas Date & Timestamp Reference Guide

### Basic Concepts

### Main Date/Time Types
- `Timestamp`: Single point in time (equivalent to Python's `datetime`)
- `DatetimeIndex`: Index of Timestamp objects
- `Period`: Time span representation
- `PeriodIndex`: Index of Period objects
- `Timedelta`: Duration (equivalent to Python's `timedelta`)
- `TimedeltaIndex`: Index of Timedelta objects

### Key Datetime Components
```python
import pandas as pd
from datetime import datetime, timedelta
```

### Creating Date/Time Objects

### From Strings
```python
# Single timestamp
ts = pd.Timestamp('2023-09-01 14:30:00')

# From various string formats
pd.to_datetime('2023-09-01')
pd.to_datetime('Sep 1, 2023')
pd.to_datetime('01/09/2023', format='%d/%m/%Y')  # Day first format
```

### From Components
```python
# Create from year, month, day components
pd.Timestamp(year=2023, month=9, day=1, hour=14, minute=30)

# Current time
pd.Timestamp.now()

# From Python datetime
pd.Timestamp(datetime(2023, 9, 1, 14, 30))
```

### Creating DatetimeIndex
```python
# From a list of datetime strings
dates = pd.DatetimeIndex(['2023-09-01', '2023-09-02', '2023-09-03'])

# Generate a range of dates
date_range = pd.date_range(start='2023-09-01', end='2023-09-30')
date_range = pd.date_range(start='2023-09-01', periods=30, freq='D')
```

### Date Ranges with Different Frequencies
```python
# Daily frequency (default)
pd.date_range('2023-09-01', periods=10, freq='D')

# Business days (Monday to Friday)
pd.date_range('2023-09-01', periods=10, freq='B')

# Weekly, Month-end, Quarter-end, Year-end
pd.date_range('2023-09-01', periods=10, freq='W')    # Weekly
pd.date_range('2023-09-01', periods=10, freq='M')    # Month end
pd.date_range('2023-09-01', periods=4, freq='Q')     # Quarter end
pd.date_range('2023-09-01', periods=3, freq='A')     # Year end

# Hours, Minutes, Seconds
pd.date_range('2023-09-01', periods=24, freq='H')    # Hourly
pd.date_range('2023-09-01', periods=60, freq='T')    # Minute
pd.date_range('2023-09-01', periods=60, freq='S')    # Second
```

### Parsing Date/Time Data

### String to Datetime Conversion
```python
# Basic conversion
pd.to_datetime('2023-09-01')

# A series of dates
pd.to_datetime(['2023-09-01', '2023-09-02', '2023-09-03'])

# Convert a dataframe column
df['date_column'] = pd.to_datetime(df['date_column'])
```

### Handling Custom Formats
```python
# Specify format with strftime codes
pd.to_datetime('01/09/2023', format='%d/%m/%Y')
pd.to_datetime('September 1, 2023', format='%B %d, %Y')

# Common format codes:
# %Y - 4-digit year (2023)
# %y - 2-digit year (23)
# %m - month as zero-padded number (01-12)
# %d - day as zero-padded number (01-31)
# %H - hour (00-23)
# %M - minute (00-59)
# %S - second (00-59)
# %B - full month name (January)
# %b - abbreviated month name (Jan)
```

### Handling Errors and Missing Values
```python
# Handle errors by coercing to NaT (Not a Time)
pd.to_datetime(['2023-09-01', 'invalid date'], errors='coerce')

# Set NaT for missing values
pd.to_datetime(['2023-09-01', None, pd.NA])
```

### Bulk Conversion Performance
```python
# Using format parameter for faster parsing when format is known
pd.to_datetime(large_series, format='%Y-%m-%d', cache=True)

# Infer format automatically (slower but convenient)
pd.to_datetime(large_series, infer_datetime_format=True)
```

### Date Attributes and Components

### Extracting Components
```python
dates = pd.date_range('2023-09-01', periods=10, freq='D')
df = pd.DataFrame({'date': dates})

# Basic components
df['year'] = df['date'].dt.year           # 2023
df['month'] = df['date'].dt.month         # 9
df['day'] = df['date'].dt.day             # 1-10
df['dayofweek'] = df['date'].dt.dayofweek # 0-6 (Monday=0)
df['dayofyear'] = df['date'].dt.dayofyear # 1-365
df['quarter'] = df['date'].dt.quarter     # 1-4
df['weekofyear'] = df['date'].dt.isocalendar().week  # ISO week number

# Time components
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['second'] = df['date'].dt.second
```

### Day Name and Month Name
```python
df['day_name'] = df['date'].dt.day_name()      # Monday, Tuesday, etc.
df['month_name'] = df['date'].dt.month_name()  # September
```

### Boolean Properties
```python
df['is_leap_year'] = df['date'].dt.is_leap_year
df['is_month_end'] = df['date'].dt.is_month_end
df['is_quarter_end'] = df['date'].dt.is_quarter_end
df['is_year_end'] = df['date'].dt.is_year_end
df['is_weekend'] = df['date'].dt.dayofweek >= 5  # Saturday=5, Sunday=6
```

### Date Math and Manipulations

### Basic Operations
```python
# Add days
df['next_day'] = df['date'] + pd.Timedelta(days=1)
df['prev_day'] = df['date'] - pd.Timedelta(days=1)

# Add various time units
df['date'] + pd.Timedelta(weeks=1)
df['date'] + pd.Timedelta(hours=3)
df['date'] + pd.Timedelta(minutes=30)
df['date'] + pd.Timedelta('3 days 2 hours')
```

### Date Offsets
```python
from pandas.tseries.offsets import Day, BusinessDay, MonthEnd, QuarterEnd, YearEnd

# Add specific offsets
df['date'] + Day(5)
df['date'] + BusinessDay(5)  # Skip weekends
df['date'] + MonthEnd(1)     # Next month end
df['date'] + QuarterEnd(1)   # Next quarter end
df['date'] + YearEnd(1)      # Next year end
```

### Date Differences
```python
# Difference between two dates
delta = df['date_column2'] - df['date_column1']  # Returns timedelta

# Convert timedelta to numeric
df['days_diff'] = (df['date_column2'] - df['date_column1']).dt.days
df['hours_diff'] = (df['date_column2'] - df['date_column1']).dt.total_seconds() / 3600
```

### Time Zones

### Setting and Converting Time Zones
```python
# Create datetime with timezone
ts_utc = pd.Timestamp('2023-09-01 14:30:00', tz='UTC')

# Convert to another timezone
ts_ny = ts_utc.tz_convert('America/New_York')

# Localize naive datetime to timezone
ts_naive = pd.Timestamp('2023-09-01 14:30:00')
ts_local = ts_naive.tz_localize('Europe/London')

# Convert timezone of a Series
df['date_utc'] = df['date'].dt.tz_localize('UTC')
df['date_local'] = df['date_utc'].dt.tz_convert('US/Pacific')
```

### Working with Different Time Zones
```python
# Common time zones
time_zones = ['UTC', 'US/Eastern', 'US/Pacific', 'Europe/London', 'Asia/Tokyo']

# Get list of all time zones
import pytz
all_timezones = pytz.all_timezones

# Remove timezone information
df['date_naive'] = df['date_with_tz'].dt.tz_localize(None)
```

### Resampling Time Series Data

### Frequency Upsampling
```python
# Starting with daily data
daily_data = pd.DataFrame({
    'date': pd.date_range('2023-09-01', periods=10, freq='D'),
    'value': range(10)
}).set_index('date')

# Upsample to hourly (creates NaN values)
hourly_data = daily_data.resample('H').asfreq()

# Fill with various methods
hourly_ffill = daily_data.resample('H').ffill()   # Forward fill
hourly_bfill = daily_data.resample('H').bfill()   # Backward fill
hourly_interp = daily_data.resample('H').interpolate()  # Linear interpolation
```

### Frequency Downsampling with Aggregation
```python
# Starting with hourly data
hourly_data = pd.DataFrame({
    'date': pd.date_range('2023-09-01', periods=240, freq='H'),
    'value': range(240)
}).set_index('date')

# Downsample with common aggregations
daily_mean = hourly_data.resample('D').mean()
daily_sum = hourly_data.resample('D').sum()
daily_max = hourly_data.resample('D').max()
daily_min = hourly_data.resample('D').min()
daily_first = hourly_data.resample('D').first()
daily_last = hourly_data.resample('D').last()
```

### Common Resampling Frequencies
```python
# Minute, Hour, Day
df.resample('T').mean()  # Minute
df.resample('H').mean()  # Hour
df.resample('D').mean()  # Day

# Week, Month, Quarter, Year
df.resample('W').mean()  # Week (ends on Sunday)
df.resample('W-MON').mean()  # Week ending Monday
df.resample('M').mean()  # Month end
df.resample('Q').mean()  # Quarter end
df.resample('A').mean()  # Year end

# Business day
df.resample('B').mean()  # Business day
```

### Time Series Analysis

### Rolling Window Operations
```python
# Set datetime as index for time series operations
df = df.set_index('date')

# Rolling windows
df.rolling(window='7D').mean()    # 7-day rolling average
df.rolling(window='30D').sum()    # 30-day rolling sum
df.rolling(window='90D').std()    # 90-day rolling standard deviation

# With minimum periods
df.rolling(window='7D', min_periods=3).mean()
```

### Expanding Window
```python
# Expanding (cumulative) window
df.expanding().mean()   # Cumulative average
df.expanding().sum()    # Cumulative sum
```

### Shifting and Lagging
```python
# Shift (lag) values by time periods
df['prev_day'] = df['value'].shift(1)      # Previous day
df['next_day'] = df['value'].shift(-1)     # Next day
df['prev_week'] = df['value'].shift(7)     # Previous week

# Shift by actual time frequency
df['prev_month'] = df['value'].shift(freq='M')  # Previous month end
```

### Period-over-Period Comparisons
```python
# Calculate changes
df['daily_change'] = df['value'].diff(1)             # Day-over-day change
df['weekly_change'] = df['value'].diff(7)            # Week-over-week change
df['monthly_change'] = df['value'].diff(30)          # Approx. month-over-month
df['yearly_change'] = df['value'].diff(365)          # Year-over-year change

# Calculate percentage changes
df['daily_pct_change'] = df['value'].pct_change(1)   # Day-over-day % change
df['weekly_pct_change'] = df['value'].pct_change(7)  # Week-over-week % change
```

### Advanced Techniques

### Working with Holidays
```python
from pandas.tseries.holiday import USFederalHolidayCalendar

# Create a calendar
cal = USFederalHolidayCalendar()

# Get holidays for a date range
holidays = cal.holidays(start='2023-01-01', end='2023-12-31')

# Create business day calendar excluding holidays
from pandas.tseries.offsets import CustomBusinessDay
business_day = CustomBusinessDay(calendar=cal)

# Generate business days
pd.date_range(start='2023-09-01', end='2023-09-30', freq=business_day)

# Check if date is a holiday
df['is_holiday'] = df.index.isin(holidays)
```

### Custom Date Offsets
```python
# Create custom business day
weekmask = 'Mon Tue Wed Thu'  # 4-day work week
holidays = ['2023-09-04', '2023-11-23']  # Labor Day and Thanksgiving
custom_bd = pd.offsets.CustomBusinessDay(holidays=holidays, weekmask=weekmask)

# Generate custom business days
pd.date_range(start='2023-09-01', end='2023-09-15', freq=custom_bd)
```

### Working with Fiscal Years
```python
# Create fiscal year with different start
from pandas.tseries.offsets import FY5253Quarter

# FY starts in July
fiscal_quarter = FY5253Quarter(startingMonth=7, qtr_with_extra_week=1, weekday=6)

# Generate fiscal quarters
pd.date_range('2023-01-01', periods=8, freq=fiscal_quarter)
```

### Irregular Time Series
```python
# Handling irregular time series
irregular_dates = pd.DatetimeIndex(['2023-09-01 10:00', '2023-09-01 10:05', 
                                    '2023-09-01 10:12', '2023-09-01 10:18'])
irregular_values = [10, 12, 15, 11]

# Create irregular series
irr_series = pd.Series(irregular_values, index=irregular_dates)

# Resample to regular 5-min intervals
regular_series = irr_series.resample('5T').mean()

# Interpolate missing values
filled_series = regular_series.interpolate(method='time')
```

### Best Practices

### General Best Practices
1. **Set datetime as index** when working with time series data for easier operations
   ```python
   df = df.set_index('date')
   ```

2. **Use timezone-aware datetimes** consistently throughout your analysis
   ```python
   df['date'] = pd.to_datetime(df['date']).dt.tz_localize('UTC')
   ```

3. **Specify formats explicitly** when parsing dates for better performance
   ```python
   pd.to_datetime(df['date'], format='%Y-%m-%d')
   ```

4. **Use NaT (Not a Time)** for missing dates rather than None or



## References
The pandas documentation can be found on: [https://pandas.pydata.org/docs/index.html](https://pandas.pydata.org/docs/index.html)
