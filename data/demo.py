import numpy as np

store_data_clean = np.random.randint(1, 10, (3, 3, 2))
print(store_data_clean.shape)
store_data_clean.resize((2, 3, 2))
print(store_data_clean.shape)
