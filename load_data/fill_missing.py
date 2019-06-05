import pandas as pd
import os
import json

# settings
FILL_LIMIT = 1 # in hours
MIN_DAYS = 100 # minimum complete days per household
test = False
fill = False
combine_daily = True

project_dir = os.path.dirname(os.getcwd())
# print(project_dir)
out_path = os.path.join(project_dir, "data", "pecan")

# parameters
resolution = 15
col = ['use']
col_names = "_".join(sorted(col)) if col else "all_col"
data_dir = "{}min_".format(resolution) + col_names
data_path = os.path.join(out_path, data_dir)
house_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
house_ids = [str(x) for x in sorted([int(x.split(".")[0]) for x in house_files])]

filled_dir = "filled_{}min_{}".format(60, col_names)
filled_path = os.path.join(out_path, filled_dir)

daily_dir = "daily_{}min_{}".format(60, col_names)
daily_path = os.path.join(out_path, daily_dir)

if test:
    dataid = house_ids[0]
    df = pd.read_csv(os.path.join(data_path, "{}.csv".format(dataid)),
                     index_col=0, parse_dates=True)
    del df.index.name
    # df = df.iloc[::-1] # sort
    df.sort_index(inplace=True)
    print(dataid)
    print(len(df))
    print(df['use'].isna().sum())
    print(df)

    ## option 1
    # idx = pd.date_range(start='1/1/2018', end='1/02/2018', freq='15min')
    # print(idx)
    # with open(os.path.join(out_path, "date-ranges_{}.json".format(data_dir)), 'r') as f:
    #     date_ranges = json.load(f)
    # dr = date_ranges[dataid]
    # idx = pd.date_range(start=dr[0], end=dr[1], freq='15min')
    # print(len(idx))
    # df.sort_index(inplace=True)
    # df = df.reindex(idx)

    # option 2
    df = df.resample('15min').mean() # also handles duplicate indexes!
    # df = df.resample('15min').asfreq()
    # df = df.asfreq(freq='15min')

    print(len(df))
    print(df['use'].isna().sum())
    print(df)

    df.interpolate(method='linear', axis=0, limit=FILL_LIMIT, inplace=True)
    print(len(df))
    print(df['use'].isna().sum())
    print(df)

if fill:
    if not os.path.exists(filled_path):
        os.makedirs(filled_path)
    print(len(house_ids))
    for i, dataid in enumerate(house_ids):
        print(i, dataid)
        df = pd.read_csv(os.path.join(data_path, "{}.csv".format(dataid)), index_col=0, parse_dates=True)
        # na_orig = df['use'].isna().sum()
        df = df.resample('H').mean()
        # na_resample = df['use'].isna().sum()
        df.interpolate(method='linear', axis=0, limit=FILL_LIMIT, inplace=True)
        # na_fill = df['use'].isna().sum()
        # print("NA orig, resample, fill", na_orig, na_resample, na_fill)
        df.to_csv(os.path.join(filled_path, "{}.csv".format(dataid)))
        # if int(dataid) > 80:
        #     break


if combine_daily:
    if not os.path.exists(daily_path):
        os.makedirs(daily_path)

    nan_in_day = {}
    df_combined = []

    print(len(house_ids))
    for i, dataid in enumerate(house_ids):
        print(i+1, dataid)
        df = pd.read_csv(os.path.join(filled_path, "{}.csv".format(dataid)), index_col=0, parse_dates=True)
        df = df.resample('D').agg(list)
        df = df[df['use'].apply(lambda x: len(x) == 24).tolist()]
        df = pd.DataFrame(df['use'].tolist(), index=df.index, columns=["h{}".format(x) for x in range(1, 25)])
        na_days = df.isna().any(axis=1).astype(int).tolist()
        complete_days = [bool(1 - x) for x in na_days]
        # only keep households with at least MIN_DAYS complete days
        if sum(complete_days) >= MIN_DAYS:
            nan_in_day[dataid] = {}
            nan_in_day[dataid]['nans'] = na_days
            nan_in_day[dataid]['date'] = df.index.map(lambda x: str(x)[:10]).tolist()
            df['dataid'] = dataid
            df.to_csv(os.path.join(daily_path, "{}.csv".format(dataid)))
            # remove days with nans before joining
            df_combined.append(df[complete_days])

    # save combined
    df_comb = pd.concat(df_combined)
    df_comb.to_csv(os.path.join(out_path, "combined_{}min_{}".format(60, col_names)))

    # write nan summaries
    with open(os.path.join(out_path, "nans_{}min_{}.json".format(60, col_names)), 'w') as fo:
        json.dump(nan_in_day, fo)
