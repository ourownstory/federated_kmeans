import os
import pandas as pd

import load_ps_data
import pecan_cred as cred

project_dir = os.path.dirname(os.getcwd())
out_path = os.path.join(project_dir, "data", "pecan")

# parameters
resolution = 15
col = ['use']
# date_limits = ('01-01-2015 00:00:00', '12-31-2015 23:59:00')
meta = pd.read_csv(os.path.join(out_path, "dataport-metadata.csv"), index_col=False)
house_ids = list(meta['dataid'].values)
# house_ids = [5785]
print("Num IDs: ", len(house_ids))

# set up connection
engine = load_ps_data.create_engine(
    user_name=cred.user_name,
    password=cred.password,
    host='dataport.pecanstreet.org',
    port=5434,
    db='postgres'
)
con = engine.connect()

col_names = "_".join(sorted(col)) if col else "all_col"
data_dir = "{}min_".format(resolution) + col_names
data_path = os.path.join(out_path, data_dir)

for i, dataid in enumerate(house_ids):
    print("{} Querrying houseid: {}".format(i, dataid))
    # run query
    df = load_ps_data.load_query(
        con=con,
        dataid=dataid,
        res=resolution,
        col=col,
        # date_limits=date_limits,
    )
    if len(df) > 0:
        # save
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        df.to_csv(os.path.join(data_path, "{}.csv".format(dataid)), index=False)
    print(i, " len: ", len(df))

con.close()


