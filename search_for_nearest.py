import netCDF4 as nc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
f=nc.Dataset("/stu01/chenss22/data_downscaling/MERITDEM_parameter/MERITDEM_ncfile/MERITDEM_height_tp_area.nc","r")
lat=f.variables["lat"][:]
lon=f.variables["lon"][:]
elv_m=f.variables["hgt"][:]
station_num=25
lat_str=list(range(station_num))
lon_str=list(range(station_num))
'''
if os.path.exists("MERITDEM_lat.txt"):
 os.remove("MERITDEM_lat.txt")
else:
 with open("MERITDEM_lat.txt","a")as f:
  for i in range(len(lat)):
   lat_str=str(lat[i])
   f.write(lat_str)
f.close()
if os.path.exists("MERITDEM_lon.txt"):
 os.remove("MERITDEM_lon.txt")
else:
 with open("MERITDEM_lon.txt","a")as f:
  for i in range(len(lon)):
   lon_str=str(lon[i])
   f.write(lon_str+)
f.close()
'''
#[LON,LAT]=np.meshgrid(lon,lat)
sta=pd.read_excel("gd_station.xlsx",header=0)
sta_lat=sta["lat"]
sta_lon=sta["lon"]
station_total=sta["station"]
f_sta=nc.Dataset("201801_stationinfo.nc","r")
elv_sta=f_sta.variables["elv"][:,0]
print(elv_sta)
#降水站点获取，区域范围：22-24，112-114
lat_tp=np.where((sta_lat>=22.0)&(sta_lat<=24.0))
lon_tp=np.where((sta_lon>=112.0)&(sta_lon<=114.0))
print(lat_tp)
sta_tp_Index=np.intersect1d(lat_tp,lon_tp)
print(sta_tp_Index)
dic={"station":station_total[sta_tp_Index],"lat":sta_lat[sta_tp_Index],"lon":sta_lon[sta_tp_Index],"elv":elv_sta[sta_tp_Index]}
dic=pd.DataFrame(dic)
dic.to_excel("tp_station.xlsx")
#new
sta=pd.read_excel("tp_station.xlsx",header=0)
sta_lat=sta["lat"]
sta_lon=sta["lon"]
station_total=sta["station"]
elv_sta=sta["elv"]

print(station_total)
for i in range(len(sta_lat)):
    distance_lat=[(lat[j]-sta_lat[i])*(lat[j]-sta_lat[i]) for j in range(len(lat))]
    lat_min=distance_lat.index(min(distance_lat))
    lat_str[i]=lat_min
for i in range(len(sta_lon)):
    distance_lon=[(lon[j]-sta_lon[i])*(lon[j]-sta_lon[i]) for j in range(len(lon))]
    lon_str[i]=distance_lon.index(min(distance_lon))

data={"station":sta["station"],"lat":lat[lat_str],"lon":lon[lon_str],"elv_grid":elv_m[lat_str,lon_str],"elv_station":elv_sta}
print(data)
data=pd.DataFrame(data)
data.to_excel("/stu01/chenss22/downscaling/station_validation/tp_nearestgrid.xlsx")
index={"lat_index":lat_str,"lon_index":lon_str}
index=pd.DataFrame(index)
index.to_excel("/stu01/chenss22/downscaling/station_validation/tp_nearestgrid_latlon_index.xlsx")
#for i in range(len(elv_sta[0,:])):
 #   if list(elv_sta[:,i].isnull()).any():
  #      continue
   # else:
    #    ELV=elv_sta[:,i]
     #   break
'''
elv_diff=data["elv"]-elv_sta[:,0]
print(elv_diff,max(elv_diff),min(elv_diff))
plt.scatter(sta_lon,sta_lat,s=20,c=elv_diff,cmap='jet')
plt.colorbar()
plt.savefig(f'/stu01/chenss22/downscaling/station_validation/elevation_diff.png')
'''
