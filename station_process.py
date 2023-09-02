import pandas as pd
from datetime import datetime
import subprocess
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
def YMD_todatetime(ds1):#读取文件名中的时间并转化datetime
 #ds=subprocess.run(f"ls /{year_month}/*.dat")
 #ds1=ds.split("\n")
 print(ds1[1][12:20])
 time_pd=ds1
 for i in range(len(ds1)):
  time=int(ds1[i][12:20])
  year=time//1000000
  month=(time-year*1000000)//10000
  day=(time-year*1000000-month*10000)//100
  hour=time-year*1000000-month*10000-day*100
  year=year+2000
  time_pd[i]=datetime(year,month,day,hour)
 return pd.to_datetime(time_pd)
def PreProcess(df_t,year_month):#在每一个dataframe中插入时间变量
 timeslt=[str(df_t["id"][i]) for i in range(len(df_t))]
 for i in range(df_t.shape[0]):
  timeslt[i]=year_month
 time=YMD_todatetime(timeslt)
 print(type(time))
 df_t.insert(1,'Date',time)
 df_t['Date']=time
 for key,values in df_t.items():
  df_t[key][df_t[key]==-888888.0000]=np.nan #缺失值
 return df_t

if __name__ == '__main__':
 os.chdir("/stu01/chenss22/downscaling/station_validation/OBS/")
 year_month=subprocess.getstatusoutput("ls")
 print(year_month)
 year_month=year_month[1]
 year_month=year_month.split("\n")
 #print(year_month)
 df=pd.DataFrame()
 latlon=pd.read_excel("/stu01/chenss22/downscaling/station_validation/gd_station.xlsx",header=0)
 #for time in year_month:
 time="201808"
 ds=subprocess.getstatusoutput(f"ls {time}/*.dat")[1]
 ds1=ds.split("\n")
 ds1=[ds1[i].strip("\n") for i in range(len(ds1))]
 print(ds1)
 print(len(ds1))
 for ds2 in ds1:
  path=f"{ds2}"
  col=["id","lat","lon","elv","ps","r6","r24","psl","u","v","t","td","rh"]
  data=pd.read_table(path,skiprows=0,header=0,sep="\s+",names=col,engine="python")
  data=PreProcess(data,ds2)
  data["lat"]=latlon["lat"]
  data["lon"]=latlon["lon"]
  data['id']=latlon["station"]
  print(data)
  df=df.append(data)
  ds_merge=xr.Dataset()
 stas=latlon["station"]
 print("id="+f"{stas}")
 n=0
 for s in stas:
   n=n+1 
   print(f'\r{n}',end=' ')
   df_s=df[df['id']==s]
   df_s.index=df_s['Date']
   df_s=df_s.sort_index()
   ds_1=xr.Dataset(data_vars=df_s.iloc[:,1:])
   ds_1=xr.concat([ds_1],pd.Index([s],name='station'))#增加一个维度
   ds_merge=xr.merge([ds_merge,ds_1])
 #为Dataset补充地理信息
 print(ds_merge)
 #ds_merge['id']=(('station'),stas) 
  #ds_merge['lon']=(('station'),ds["lon"])
  #ds_merge['lat']=(('station'),ds["lat"])
  #ds_merge['elev']=(('station'),ds["elv"])
 os.system(f"rm -f /stu01/chenss22/downscaling/station_validation/{time}_stationinfo.nc")
 ds_merge.to_netcdf(f'/stu01/chenss22/downscaling/station_validation/{time}_stationinfo.nc')
 ncfil=xr.open_dataset(f'/stu01/chenss22/downscaling/station_validation/{time}_stationinfo.nc')
 elev=ncfil.elv
 plt.scatter(ncfil.lon,ncfil.lat,s=6,c=elev,cmap='jet')
 plt.colorbar()
 plt.savefig(f'/stu01/chenss22/downscaling/station_validation/{time}_elevation.png')


