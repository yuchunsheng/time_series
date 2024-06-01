import pandas as pd
import ast
import datetime
import matplotlib.pyplot as plt

# Function to convert string to dictionary
def str_to_dict(s):
    pairs = s.split(', ')
    d = {}
    for pair in pairs:
        key, value = pair.split(':', 1)  # split at the first colon
        if value.startswith('{') and value.endswith('}'):
            value = ast.literal_eval(value)
        if key == 'operationTime':
            value = pd.to_datetime(datetime.datetime.fromtimestamp(int(value) / 1000).strftime('%Y-%m-%d %H:%M:%S'))
        d[key] = value
    return d

# Read the file line by line, convert each line to a dictionary, and append to a list
data = []
with open('sensor3.txt', 'r') as file:
    for line in file:
        line = line.rstrip('\\n')
        d = str_to_dict(line)
        data.append(d)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Assuming 'df' is your DataFrame
df['action'] = df['action'].apply(lambda x: 1 if x['OnOff'] == 'ON' else 0)
df['type'] = df['type'].str.replace('\n', '')

# Now you can work with the DataFrame 'df'
print(df.head())

# Assuming 'df' is your DataFrame
mac_values = df['mac'].unique()

sensor_dataframes = {}
for mac in mac_values:
    sensor_dataframes[mac] = df[df['mac'] == mac]

# print(sensor_dataframes['282C02BFFFED7102'])
# Now 'dataframes' is a dictionary where each key is a unique MAC address
# and each value is a DataFrame containing only the rows with that MAC address
one_sensor = sensor_dataframes['282C02BFFFED7102']
one_sensor.head()

# # Assuming df is your DataFrame and it has been defined earlier
# df = pd.DataFrame({
#     'action': [0, 1, 0, 1],
#     'operationTime': ['2024-03-23 11:29:13', '2024-03-23 11:29:10', '2024-03-23 11:28:35', '2024-03-23 11:28:30'],
#     'mac': ['282C02BFFFED7102', '282C02BFFFED7102', '282C02BFFFED7102', '282C02BFFFED7102'],
#     'type': ['SensorPir', 'SensorPir', 'SensorPir', 'SensorPir']
# })

# Print the keys (column names) of the DataFrame
print(one_sensor.keys())
one_sensor.info()

# Convert operationTime to datetime
# one_sensor['operationTime'] = pd.to_datetime(one_sensor['operationTime'])

# Set operationTime as the index
one_sensor.set_index('operationTime', inplace=True)

# Resample the DataFrame by hour and sum the action column
df_resampled = one_sensor.resample('h')['action'].sum()

# Convert the index to datetime
# df_resampled.index = pd.to_datetime(df_resampled.index)
print(type(df_resampled))
# print(df_resampled)
# Convert series to dataframe
df = df_resampled.to_frame()

# Reset the index of the dataframe
df.reset_index(inplace=True)

# Rename the old index column
df.rename(columns={'index': 'operationTime'}, inplace=True)

# Add new column to dataframe
# df['index'] = range(1, len(df) + 1)
df['unique_id'] = '282C02BFFFED7102'

# Rename the columns
df = df.rename(columns={"operationTime": "ds", "action": "y"})

print(df.head())

df.to_csv('282C02BFFFED7102.csv', index=False)

# # Plot the time series
# plt.figure(figsize=(10, 6))
# plt.plot(df_resampled)
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Time Series Plot')
# plt.grid(True)
# plt.show()