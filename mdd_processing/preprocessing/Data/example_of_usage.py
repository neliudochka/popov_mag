import pandas as pd
from .Data import default_config, Data
from .DataConfig import DataConfig, TimestampType

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


file_path = "../../../tool/files_for_test/20250509_igor/084539/20250509_084539_MD-R3_#0112_RF.csv"
final_conf_path = "data_configs/final.yaml"
own_conf_path = "data_configs/own.yaml"
mila_conf_path = "data_configs/test.yaml"
final_dc = DataConfig.from_yaml(final_conf_path)
mila = DataConfig.from_yaml(mila_conf_path)
print(final_dc)
cf = DataConfig.from_yaml(mila_conf_path)
bc = DataConfig.from_yaml(default_config)
owc = DataConfig.from_yaml(own_conf_path)
data = Data.from_file(file_path, owc)
print(data.df)
# # print(build_column_rename_map(data.column_names, gen_column_names(DataConfig.from_yaml(default_config))))
print(data.column_names.yaw_col)
data.replace_config(cf)
data.save(new_filepath=file_path+"pupu")
datanew = Data.from_file(file_path+"pupu", cf)
print(datanew.df)
gnss = "../../../tool/files_for_test/20250325/rtk.csv"
g = "data_configs/external_gnss.yaml"
cf = DataConfig.from_yaml(g)
d = Data.from_file(gnss, cf)
print(d.df)
print(d.df[d.column_names.utc_timestamp_col])
print(d.column_names)
final_d = Data.from_empty(config=final_dc)
print(data.config.columns['timestamp_col'].type == TimestampType.DAY)