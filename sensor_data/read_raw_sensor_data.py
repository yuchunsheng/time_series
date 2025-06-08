import pandas as pd
import os

# Define the path to the data file
# Assuming the script is in D:\deeplearnig_data\PIR_sensor_data\csh101
# If the script is elsewhere, you'll need to provide the full absolute path to the .txt file
file_path = r"D:\deeplearnig_data\PIR_sensor_data\csh101\csh101.rawdata.txt"

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
        # Try reading with a common delimiter (e.g., comma, space, tab).
        # You might need to adjust the 'sep' parameter based on your file's structure.
        # For example, for tab-separated, use sep='\t'.
        # If there's no header row, you can add header=None.
        df = pd.read_csv(path_to_file, sep=r'\s+', engine='python') # Example: assumes space/tab delimited
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
        print("\nFirst 5 rows of the DataFrame:")
        print(sensor_data_df.head())

        print("\nDataFrame Info:")
        sensor_data_df.info()