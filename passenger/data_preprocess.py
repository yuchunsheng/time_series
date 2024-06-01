import os
import pandas as pd

path='c:\\temp\\python_play_ground\\sensor\\archive'
order_items = pd.read_csv(os.path.join(path, 'olist_order_items_dataset.csv'))
orders = pd.read_csv(os.path.join(path, 'olist_orders_dataset.csv'))
orders = orders[['order_id', 'order_purchase_timestamp']]
data = pd.merge(order_items, orders, on='order_id')