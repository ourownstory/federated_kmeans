import json
import pandas as pd
import os
from os.path import isfile, join
import datetime
import matplotlib.pyplot as plt


project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
print(project_dir)
out_path = os.path.join(project_dir, "data", "pecan")

compute_date_ranges = False

# parameters
resolution = 15
col = ['use']
col_names = "_".join(sorted(col)) if col else "all_col"
data_dir = "{}min_".format(resolution) + col_names
data_path = os.path.join(out_path, data_dir)
house_files = [f for f in os.listdir(data_path) if isfile(join(data_path, f))]
house_ids = [x.split(".")[0] for x in house_files]

# takes long
# def get_min_max_date(dates):
#     dates = dates.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
#     min_date = dates.min().strftime("%Y-%m-%d %H:%M:%S")
#     max_date = dates.max().strftime("%Y-%m-%d %H:%M:%S")
#     return min_date, max_date

if compute_date_ranges:
    date_ranges = {}
    print(len(house_ids))
    for i, dataid in enumerate(house_ids):
        print(i)
        df = pd.read_csv(os.path.join(data_path, "{}.csv".format(dataid)), index_col=False)
        date_ranges[str(dataid)] = (df["localtime"][len(df)-1], df["localtime"][0])

    with open(os.path.join(out_path, "date-ranges_{}.json".format(data_dir)), 'w') as fo:
        json.dump(date_ranges, fo)
else:
    with open(os.path.join(out_path, "date-ranges_{}.json".format(data_dir)), 'r') as f:
        date_ranges = json.load(f)

print(date_ranges)


def plot_date_range(dates):
    for i, (id, dr) in enumerate(dates.items()):
        dr = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dr]
        plt.plot(dr, [i, i], 'b')
    plt.show()

plot_date_range(date_ranges)
