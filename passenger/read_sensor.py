import datetime
import ast

def str_to_dict(s):
    pairs = s.split(', ')
    d = {}
    for pair in pairs:
        key, value = pair.split(':', 1)  # split at the first colon
        if value.startswith('{') and value.endswith('}'):
            value = ast.literal_eval(value)
        if key == 'operationTime':
            value = datetime.datetime.fromtimestamp(int(value) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        d[key] = value
    return d

with open('sensor1.txt', 'r') as file:
    for line in file:
        line = line.rstrip('\\n')
        d = str_to_dict(line)
        print(d)
