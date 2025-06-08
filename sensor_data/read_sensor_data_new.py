import pandas as pd
import os
import io
# Define the path to the data file
# Assuming the script is in D:\deeplearnig_data\PIR_sensor_data\csh101
# If the script is elsewhere, you'll need to provide the full absolute path to the .txt file
file_path = r"D:\deeplearnig_data\PIR_sensor_data\csh102\csh102.rawdata.txt"
# In a real-world scenario, you would load your data from a file, for example:
# df = pd.read_csv('your_sensor_data.csv')

# # For this example, we'll recreate a sample of your DataFrame from the provided text.
# # Note: The original 'Timestamp' column seems to be a combination of 'Date' and 'Time'.
# # We will assume a single 'Timestamp' column exists for simplicity, as shown in your example output.
# data = """Timestamp,SensorID,Attribute1,Attribute2,Value,SensorType
# 2011-06-15 00:06:32.834414,M021,Bedroom,Bed,ON,Control4-Motion
# 2011-06-15 00:06:33.988964,M021,Bedroom,Bed,OFF,Control4-Motion
# 2011-06-15 00:15:01.957718,LS013,Ignore,Ignore,6,Control4-LightSensor
# 2011-06-15 00:25:01.892474,LS013,Ignore,Ignore,7,Control4-LightSensor
# 2011-06-15 03:37:46.585185,M021,Bedroom,Bed,ON,Control4-Motion
# """

# # Use io.StringIO to read the string data as if it were a file
# df = pd.read_csv(io.StringIO(data))

# # Ensure the 'Timestamp' column is treated as a proper datetime object,
# # which is crucial for time-based indexing and sorting.
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])


def load_data_to_dataframe(path_to_file):
    """
    Reads a text file into a Pandas DataFrame.

    Args:
        path_to_file (str): The full path to the text file.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the file,
                          or None if an error occurs.
    """
    try:
        # Define column names based on the observed data structure
        column_names = [
            'Date', 'Time', 'SensorID',
            'Attribute1', 'Attribute2', 'Value', 'SensorType'
        ]

        # Read the space-separated file, with no header row,
        # assign defined column names, and parse Date and Time columns as a single Timestamp.
        df = pd.read_csv(
            path_to_file,
            sep=r'\s+',             # Delimiter is one or more whitespace characters
            header=None,            # No header row in the file
            names=column_names,     # Assign our defined column names
            # parse_dates={'Timestamp': ['Date', 'Time']}, # Deprecated way for nested sequences
            engine='python'         # Use the Python parsing engine for regex separator
        )

        # Combine 'Date' and 'Time' columns into a single 'Timestamp' column
        # and convert it to datetime objects.
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='mixed')

        # Optionally, drop the original 'Date' and 'Time' columns if they are no longer needed
        df.drop(columns=['Date', 'Time'], inplace=True)

        print(f"Successfully loaded data from: {path_to_file}")

        return df
    except FileNotFoundError:
        print(f"Error: The file was not found at {path_to_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

if __name__ == "__main__":
    sensor_data_df = load_data_to_dataframe(file_path)

    if sensor_data_df is not None:
        # print("\nFirst 5 rows of the DataFrame:")
        # print(sensor_data_df.head())

        # print("\nDataFrame Info:")
        # sensor_data_df.info()
        # --- The Pivot Operation ---
        # We will use the .pivot_table() method from pandas.
        # - index: The column to use for the new DataFrame's rows.
        # - columns: The column whose unique values will become the new DataFrame's columns.
        # - values: The column whose data will populate the cells of the new DataFrame.
        # - aggfunc: Specifies how to handle duplicate index/column pairs. 'first' takes the first occurring value.
        pivot_df = sensor_data_df.pivot_table(
            index='Timestamp',
            columns='SensorType',
            values='Value',
            aggfunc='first'
        )

        # If your original DataFrame truly has separate 'Date' and 'Time' columns,
        # you would first need to combine them into a 'Timestamp' column like this:
        #
        # from your_column_names import column_names
        # df.columns = column_names # Assuming the raw file has no headers
        # df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        #
        # Then you would run the same pivot_table() function on the modified df.


        # --- Display the results ---
        print("------ Original DataFrame ------")
        print(sensor_data_df)
        print("\n" + "="*50 + "\n")
        print("------ Pivoted DataFrame ------")
        print(pivot_df)