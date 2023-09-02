import os
import pandas as pd
import subprocess
import netCDF4 as nc
from datetime import datetime
import xarray as xr
import numpy as np
"""
将站点文件转换为nc文件,维度（stationID,date）
stationID：站点号
var_list：需要处理的数据文件名称
lat_index：站点

"""
def write_to_nc(stationID,var_list,lat_index,lon_index):
 df_total=pd.DataFrame()
 time=stationID[:]
 station_id=stationID[:]
 xds=xr.Dataset()
 #获取每一小时所有最邻近格点变量信息，插入时间信息
 for i in range(24):
  dict_total={}
  dict_total["stationID"]=station_id
  for var_fil in var_list:
   f_mo=nc.Dataset(var_fil,"r")
   varname=var_fil.split("_")[-3]
   year=int(var_fil.split("_")[3])
   month=int(var_fil.split("_")[4])
   day=int(var_fil.split("_")[5])
   date=datetime(year,month,day,i)
   var_nc_01=f_mo[varname][i,:,:]
   var_seq=var_nc_01[lat_index,lon_index]
   print(var_seq)
   var_dict={}
   var_dict[varname]=var_seq
   dict_total.update(var_dict)
   f_mo.close()
  dict_total=pd.DataFrame(dict_total)
  #在dataframe中插入时间变量
  for k in range(len(stationID)):
   print(k)
   time[k]=datetime(year,month,day,int(i))
   print(station_id)
  dict_total.insert(1,'Date',time)
  dict_total["Date"]=time
  print(dict_total)
  #连接dataframe
  #df_total=df_total.append(dict_total)
  df_total=pd.concat([df_total,dict_total])
  
  #生成dataset,每个变量是station,date的二维数据,利用xr.merge函数聚合
 for s in stationID:
  index=df_total["stationID"]==s
  pra=df_total[index]
  pra.index=pra["Date"] 
  pra=pra.sort_index()  
  pra=xr.Dataset(pra.iloc[:,1:])
  pra.fillna(-888888.0000)
  pra=xr.concat([pra],pd.Index([s],name="station"))#创建一个维度，index内容为s,维度名字叫station
  xds=xr.merge([xds,pra])  
 print(xds)
 longname='ERA5LAND_GD_fine_tp_mei_ZH_' 
 xds.to_netcdf(f"station_data/{longname}{month}_{day}.nc")

if __name__ == '__main__':
#根据经纬度查找最近格点在数组中的位置
 month=["01"]#,"08"]
 days=list(range(1,32,1))
 days=[str(days[i]).zfill(2) for i in range(len(days))]#将days变成占两位的字符串
 f=nc.Dataset("/stu01/chenss22/data_downscaling/MERITDEM_parameter/MERITDEM_ncfile/MERITDEM_height.nc","r")
 lat=f.variables["lat"][:]
 lon=f.variables["lon"][:]
 latlon=pd.read_excel("tp_nearestgrid.xlsx",header=0)
 lat_grid=latlon["lat"]
 lon_grid=latlon["lon"]
 station_ID=latlon["station"]
 index=pd.read_excel("tp_nearestgrid_latlon_index.xlsx",header=0)
 lat_index=index["lat_index"]
 lon_index=index["lon_index"]

 os.chdir("fine_data_after_downscaling")
 for mo in month:
  for day in days:
   var_list=subprocess.getoutput(f"ls ERA5LAND_GD_fine_2018_{mo}_{day}_tp_mei_ZH.nc")
   var_list=var_list.split("\n")
   var_list=[var_list[i].strip("\n") for i in range(len(var_list))]
   write_to_nc(station_ID,var_list,lat_index,lon_index)
