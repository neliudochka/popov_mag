import pandas as pd
from matplotlib import pyplot as plt

from mdd_processing.preprocessing.main import get_data_with_config

external_gnss_filepath = "/home/mila/work/popov_all/15102026/20261015.csv"
DATA_EXTERNAL_GNSS_PATH = "/media/mila/WinToUSB/Users/ropuh/mila/Code/project/mdd_processing/preprocessing/Data/data_configs/external_gnss_popov.yaml"
gnss = get_data_with_config(
        external_gnss_filepath, DATA_EXTERNAL_GNSS_PATH
    )

res = pd.read_csv("/home/mila/work/popov_all/15102026/foot/файл 02_foot_const_med0_ctf0_eg.asc", delimiter="\t")
traj_gps = pd.read_csv("/media/mila/WinToUSB/Users/ropuh/mila/Code/project/mdd_processing/smth.csv", delimiter=",")
print(res)
print(gnss)
print(traj_gps)

fig, ax = plt.subplots(figsize=(10, 10))

# ax.scatter(
#     gnss.df[gnss.column_names.lon_col],
#     gnss.df[gnss.column_names.lat_col],
#     color="red",
#     label="original",
#     s=3
# )
#
# #
# traj_gps=traj_gps[traj_gps["Track ID"]==4].dropna().reset_index(drop=True)
#
# print(len(traj_gps["Track ID"]))
ax.scatter(
    traj_gps["Longitude [°]"].dropna(),
    traj_gps["Latitude [°]"].dropna(),
    color="black",
    label="result",
    s=3
)

ax.scatter(
    res["Longitude [°]"].dropna(),
    res["Latitude [°]"].dropna(),
    color="blue",
    label="traj",
    s=1
    # linewidth=1.0,
)

# traj_gps=traj_gps[traj_gps["Track ID"]==8].reset_index(drop=True)

# ax.scatter(
#     traj_gps["Longitude [°]"].dropna(),
#     traj_gps["Latitude [°]"].dropna(),
#     color="blue",
#     s=3
# )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.savefig("plt")
