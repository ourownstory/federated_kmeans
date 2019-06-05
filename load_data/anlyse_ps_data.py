import json
import pandas as pd
import numpy as np
import os
from os.path import isfile, join
import datetime
import matplotlib.pyplot as plt

project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
out_path = os.path.join(project_dir, "data", "pecan")
fig_path = os.path.join(project_dir, "data", "pecan_info")


def plot_date_range(dates):
    for i, (id, dr) in enumerate(dates.items()):
        dr = [datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dr]
        plt.plot(dr, [i, i], 'b')
    plt.show()


def compute_plot_date_ranges(
        compute_date_ranges=False,
        resolution=15,
        col=['use']
        ):
    col_names = "_".join(sorted(col)) if col else "all_col"
    data_dir = "{}min_".format(resolution) + col_names
    data_path = os.path.join(out_path, data_dir)
    house_files = [f for f in os.listdir(data_path) if isfile(join(data_path, f))]
    house_ids = [x.split(".")[0] for x in house_files]

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

    plot_date_range(date_ranges)


def plot_days_and_nans(
        resolution=60,
        col=['use'],
        plottype=None
        ):
    col_names = "_".join(sorted(col)) if col else "all_col"
    f_name = os.path.join(out_path, "nans_{}min_{}.json".format(resolution, col_names))
    with open(f_name, 'r') as f:
        nan_in_day = json.load(f)

    to_date = np.vectorize(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
    if plottype == 'heatmap':
        idx = pd.date_range(start='1/1/2012', end='12/31/2019', freq='D')
        df_all = []
    print(len(nan_in_day))
    for i, (id, date_nan_dict) in enumerate(nan_in_day.items()):
        # print(i)
        if plottype == 'heatmap':
            nans = 1 - 0.4*np.array(date_nan_dict["nans"])
            df = pd.DataFrame(
                data=nans,
                index=to_date(date_nan_dict["date"]),
                columns=['nans']
            )
            # fill with nan, append
            df = df.reindex(idx, fill_value=0)
            # make double, such that pixels are a bit wider
            df_all += [df]*4
        else:
            nans = np.array(date_nan_dict["nans"])
            dates = to_date(date_nan_dict["date"])
            not_nan_dates = dates[np.equal(nans, 0)]
            nan_dates = dates[np.equal(nans, 1)]
            plt.plot(not_nan_dates, i*np.ones(len(not_nan_dates)), 'b,', markersize=2)
            plt.plot(nan_dates, i*np.ones(len(nan_dates)), 'r,', markersize=2)
    if plottype == 'heatmap':
        df_all = pd.concat(df_all, axis=1)
        a = np.transpose(df_all.values)
        plt.imshow(a, cmap='gray', vmin=0, vmax=1)
        plt.imsave(os.path.join(fig_path, "data_validity_over_time.png"), a,
                   dpi=600, cmap='gray', vmin=0, vmax=1)
    plt.show()


def plot_complete_days_hist(
        resolution=60,
        col=['use'],
        ):
    col_names = "_".join(sorted(col)) if col else "all_col"
    f_name = os.path.join(out_path, "nans_{}min_{}.json".format(resolution, col_names))
    with open(f_name, 'r') as f:
        nan_in_day = json.load(f)

    completes = []
    print(len(nan_in_day))
    for i, (id, date_nan_dict) in enumerate(nan_in_day.items()):
        # print(i)
        complete = len(date_nan_dict["nans"]) - sum(date_nan_dict["nans"])
        completes.append(complete)

    plt.hist(completes, bins=50)
    plt.savefig(os.path.join(fig_path, "complete_days_per_household.png"), dpi=600)
    plt.show()


def plot_daily_energy_hist():
    df = pd.read_csv(os.path.join(out_path, "combined_60min_use"))
    x = df.values[:, 1:25]
    daily = np.sum(x, axis=1).astype(np.int)
    print("plotting")
    plt.hist(daily, bins=100, range=(0, 200))
    plt.savefig(os.path.join(fig_path, "daily_energy_hist_200max.png"), dpi=600)
    plt.show()


def main():
    # compute_plot_date_ranges(compute_date_ranges=False)
    # plot_days_and_nans(plottype='heatmap')
    # plot_complete_days_hist()
    plot_daily_energy_hist()


if __name__ == "__main__":
    main()

