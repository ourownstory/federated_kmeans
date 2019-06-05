import os

import pandas as pd
import numpy as np


# settings
normalize_per_household = True
TEST_SPLIT = 0.2
np.random.seed(seed=0)

# params
project_dir = os.path.dirname(os.getcwd())
print(project_dir)
out_path = os.path.join(project_dir, "data", "pecan")
train_path = os.path.join(project_dir, "data", "pecan_train")
test_path = os.path.join(project_dir, "data", "pecan_test")

df = pd.read_csv(os.path.join(out_path, "combined_60min_use"))
x = df.values[:, 1:25].astype(np.float)
house_ids = df.values[:, -1]
dates = df.values[:, 0]
unique_ids = np.unique(house_ids)

### remove outliers
daily = np.sum(x, axis=1)
below = daily < 2
above = daily > 200
outliers = np.logical_or(below, above)
keep = np.invert(outliers)

before_num = len(x)
x = x[keep]
house_ids = house_ids[keep]
dates = dates[keep]
daily = daily[keep]
print("outliers removed: ", before_num - len(x))

if normalize_per_household:
    for id in unique_ids:
        house_avg = np.mean(x[house_ids == id])
        x[house_ids == id] = x[house_ids == id] / house_avg
else:
    ### normalize by daily avg
    x = x / np.expand_dims(daily, axis=1)
    x *= x.shape[1]
print("normalized households")

# round to 5 decimals:
x = np.around(x.astype(np.float), decimals=5)

# split data
total_ids_num = len(unique_ids)
test_ids_num = int(TEST_SPLIT*total_ids_num)
test_ids = np.random.choice(unique_ids, size=test_ids_num, replace=False)
train_ids = list(set(unique_ids) - set(test_ids))

# print(len(unique_ids))
# print(len(train_ids))
# print(len(test_ids))

test_mask = np.isin(house_ids, test_ids)
train_mask = np.invert(test_mask)

x_train = x[train_mask, :]
x_test = x[test_mask, :]
house_ids_train = house_ids[train_mask]
house_ids_test = house_ids[test_mask]
dates_train = dates[train_mask]
dates_test = dates[test_mask]

# write
print("writing data")
pd.DataFrame(x_train).to_csv(os.path.join(train_path, "x.csv"), index=False, header=False)
pd.DataFrame(house_ids_train).to_csv(os.path.join(train_path, "house_ids.csv"), index=False, header=False)
pd.DataFrame(dates_train).to_csv(os.path.join(train_path, "dates.csv"), index=False, header=False)
pd.DataFrame(x_test).to_csv(os.path.join(test_path, "x.csv"), index=False, header=False)
pd.DataFrame(house_ids_test).to_csv(os.path.join(test_path, "house_ids.csv"), index=False, header=False)
pd.DataFrame(dates_test).to_csv(os.path.join(test_path, "dates.csv"), index=False, header=False)

# print(len(unique_ids))
# print(unique_ids)
# print(house_ids)
# print(dates)

# print(x_train.shape)
# print(x_test.shape)
# print(x.shape)
