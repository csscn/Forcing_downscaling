import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import *

def rh(data_grid,index1,index_min,index_max,grid_nearest_info):
 #马古列斯函数用到的常数
 epsi=.62198 # Ratio of molecular weight of water and dry air
 abs0=-273.15
 AW1, BW1, CW1 = 611.21, 17.502, 240.97  
 AW2, BW2, CW2 = 611.21, 17.368, 238.88
 AI,  BI,  CI  = 611.15, 22.452, 272.55
##
 m=data_grid/(1-data_grid) #混合比m=q/(1-q)
 Ta=grid_nearest_info["t2m"][index1,index_min:index_max]#0:736
 p=grid_nearest_info["sp"][index1,index_min:index_max]#0:736
 A=np.ones_like(Ta)*AW1
 B=np.ones_like(Ta)*BW1
 C=np.ones_like(Ta)*CW1
 A[Ta+abs0>5],B[Ta+abs0>5],C[Ta+abs0>5]=AW2, BW2, CW2
 A[Ta+abs0<-5],B[Ta+abs0<-5],C[Ta+abs0<-5]=AI,  BI,  CI  
 es=A*np.exp(B*(Ta+abs0)/(Ta+abs0+C)) #马古列斯公式计算饱和水汽压
 ms=epsi*es/(p-es)
 data_grid=m/ms*100 #计算最邻近格点相对湿度
 return data_grid

def plot_timeslt(stationID_origin,grid_nearest_station,grid_nearest_info,grid_fine_info,station_info,month,DATE):
    #5个站点和MERITDEM相差最大的点；5个站点和MERITDEM相差最小的点；5个海拔最大的点
    ddf=pd.read_excel("/stu01/chenss22/downscaling/station_validation/nearestgrid_coarse_interp.xlsx")
    elevation_coarse=ddf["diff"]
    print(stationID_origin)
    stationID_origin=list(stationID_origin)

    elevation_fine=grid_nearest_station["elevation_diff"]
    grid_nearest_station["elevation_diff"]=np.abs(grid_nearest_station["elevation_diff"])
    station_elev=grid_nearest_station["station_elevation"]
    elevation_diff=grid_nearest_station["elevation_diff"]
    index_col=grid_nearest_station["index"]#index_col表示在原站点序列中的位置
    vars_coarrse=["d2m","Q","sp","t2m","u10","v10","tp"]
    vars_fine=["d2m","Q","sp","t2m","ws","tp"]
    var_station=["td","rh","ps","t","u","v","R6","R24"]
    units=["(k)","(%)","(hPa)","(k)","(m/s)","mm/6h"]
    grid_nearest_station.sort_values(by="station_elevation",inplace=True,ascending=False)#按照站点和格点的绝对海拔差进行排序


    #细网格降水区域的5个站点和MERITDEM相差最大的点；5个站点和MERITDEM相差最小的点；5个海拔最大的点
    df_tp=pd.read_excel("/stu01/chenss22/downscaling/station_validation/tp_nearestgrid.xlsx")
    tp_station_elev=df_tp["elv_station"]
    indexcol_tp=df_tp["index"]
    stationid_tp_origin=df_tp["station"]
    stationid_tp_origin=list(stationid_tp_origin)
    print(type(stationid_tp_origin))
    df_tp["elv_diff"]=np.abs(df_tp["elv_diff"])
    #df_tp.sort_values(by="elv_diff",inplace=True,ascending=True) #根据海拔差进行排序
    df_tp.sort_values(by="elv_station",inplace=True,ascending=False) #根据站点海拔进行排序
    stationid_tp=df_tp["station"]
    stationid_tp=list(stationid_tp)
    station_tp_min_diff=stationid_tp[0:5]
    station_tp_max_diff=stationid_tp[-1:-6]

    ####
    stationID=grid_nearest_station['stationID']
    print(type(stationID[0]))
    stationID=list(stationID)
    staid_min_diff=stationID[0:5]
    staid_max_diff=stationID[-5:-1]
    staid_max_diff.append(stationID[-1])
    print(staid_max_diff)
    #staid_min_diff=staid_max_diff
    select_sta=[59279,59074]
    for k in range(2):
      print(stationID_origin)
      valiation_index=stationID_origin.index(select_sta[k])
      #valiation_index=stationID_origin.index(staid_min_diff[k])
      print("###index")
      print(valiation_index)
      station_low_index=index_col[valiation_index]
      print(station_low_index)
      #降水
      va_tp_index=stationid_tp_origin.index(station_tp_min_diff[k])
      station_tp_index=indexcol_tp[va_tp_index]
      #print(stationID.index(59074))
      #station_high_index=index_col[stationID.index(59074)]
      #station_low_index=index_col[stationID.index(59324)]
      j=0
      line_colors = sns.color_palette("Set3", n_colors=8, desat=.85).as_hex()
      index_min=0
      index_max=743
      fig,axes=plt.subplots(3,2,sharex='col')
      i=0
      for jj in range(3):
       if i>=5:
        break
       for kk in range(2):
         j=0
         if i>=5:
           break
         else:
          print(grid_nearest_info[vars_coarrse[i]].shape)
          print(grid_fine_info[vars_fine[i]].shape)
          #模式输出降水为mm/h，站点观测为mm/6h;mm/24h
          if vars_fine[i]=="tp":
           var_info_coarse=grid_nearest_info[vars_coarrse[-1]][station_tp_index,index_min:index_max]
           var_info_fine=grid_fine_info[vars_fine[i]][station_tp_index,index_min:index_max]
           #求6h、24h累计降水
           r6_coarse=var_info_coarse.resample(time='6H').sum()
           r6_fine=var_info_fine.resample(time='6H').sum()
           r24_coarse=var_info_coarse.resample(time='24H').sum()
           r24_fine=var_info_fine.resample(time='24H').sum()


           '''
           r6_coarse,r6_fine,r24_coarse,r24_fine=[]
           for tt in range(0,744,6):
             r6_coarse=r6_coarse.append(np.nansum(var_info_coarse[tt:tt+5]))
             r6_fine=r6_fine.append(np.nansum(var_info_fine[tt:tt+5]))
             if (tt+1)%24==0:
               r24_fine=r24_fine.append(np.nansum(var_info_fine[tt:tt+23]))
               r24_coarse=r24_coarse.append(np.nansum(var_info_coarse[tt:tt+23]))
           '''
           station_info1_r6=station_info[var_station[-2]][station_tp_index,index_min:index_max]
           station_info1_r24=station_info[var_station[-1]][station_tp_index,index_min:index_max]
          else:
           var_info_coarse=grid_nearest_info[vars_coarrse[i]][station_low_index,index_min:index_max]
           var_info_fine=grid_fine_info[vars_fine[i]][station_low_index,index_min:index_max]
           station_info1=station_info[var_station[i]][station_low_index,index_min:index_max]
          if vars_coarrse[i]=="Q":
           var_info_coarse=rh(var_info_coarse,station_low_index,index_min,index_max,grid_nearest_info)
           var_info_fine=rh(var_info_fine,station_low_index,index_min,index_max,grid_fine_info)
          if vars_coarrse[i]=="sp":
           var_info_coarse=var_info_coarse/100
           var_info_fine=var_info_fine/100
          if vars_coarrse[i]=="u10":
            v=grid_nearest_info[vars_coarrse[i+1]][station_low_index,index_min:index_max]
            var_info_coarse=np.sqrt(var_info_coarse.values**2+v**2)
          if var_station[i]=="u":
            v=station_info[var_station[i+1]][station_low_index,index_min:index_max]
            station_info1=np.sqrt(station_info1.values**2+v**2)
          #plot
          #if i==4:
           # var_station[i]="ws"
          if i<5:
          
           station_info1.plot.line(x='Date', ax=axes[jj][kk],
           color='black', linewidth=1.0, linestyle="solid", alpha=0.7, label='Observed')

          #求相关系数、RMSE

          
           var_info_fine.plot.line(x='Date', ax=axes[jj][kk],
           linewidth=1.0, linestyle="solid", label=f'fine', color=line_colors[j+4])
           axes[jj][kk].legend(loc='best', shadow=False, frameon=False, fontsize=6)  # ,color=line_colors
           var_info_coarse.plot.line(x='Date', ax=axes[jj][kk],
           linewidth=1.0, linestyle="solid", label=f'coarse', color=line_colors[j+3])
           axes[jj][kk].legend(loc='best', shadow=False, frameon=False, fontsize=6)  # ,color=line_colors

           axes[jj][kk].set_xlabel('')
           if i==4:
            axes[jj][kk].set_ylabel("ws"+units[i],fontsize=6)
           else:
             axes[jj][kk].set_ylabel(var_station[i]+units[i],fontsize=6)
           axes[jj][kk].tick_params(axis='both',labelsize=6)
           if i>0&i<3:
             axes[jj][kk].set_title("")
            # axes[jj][kk].set_xticklabels(['']*9)
           elif i==0:
             axes[jj][kk].set_title("")#(f"station={staid_min_diff[k]},elevation={station_elev[station_low_index]},\
             #fine_elevation_diff={elevation_fine[station_low_index]},coarse_elevation_diff={elevation_coarse[station_low_index]}",fontsize=6)
            # axes[jj][kk].set_xticklabels(['']*9)
           else:
                axes[jj][kk].set_title("")
          else:
            station_info1_r6.plot.line(x='Date', ax=axes[jj][kk],
            color='black', linewidth=1.0, linestyle="solid", alpha=0.7, label='Observed')
            r6_coarse.plot.line(x='Date', ax=axes[jj][kk],
            linewidth=1.0, linestyle="solid", label=f'coarse', color=line_colors[j+3])
            r6_fine.plot.line(x='Date', ax=axes[jj][kk],
            linewidth=1.0, linestyle="solid", label=f'fine', color=line_colors[j+4])

          i=i+1
            # axes[jj][kk].set_xticklabels(DATE)

       j=j+1
      plt.delaxes(axes[2, 1])
      plt.suptitle(f"station={staid_min_diff[k]},elevation={station_elev[valiation_index]},\
      fine_elevation_diff={round(elevation_fine[valiation_index],2)},coarse_elevation_diff={round(elevation_coarse[valiation_index],2)}",fontsize=6)
      plt.tight_layout()
      plt.savefig(f"tqp_timeslt_{month}_{staid_min_diff[k]}_maxelevation.png",dpi=300)

#主程序
grid_nearest_station=pd.read_excel("/stu01/chenss22/downscaling/station_validation/fine_data_after_downscaling/station_data/valitation_station.xlsx")#非缺失站点的信息
station_info=xr.open_dataset("/stu01/chenss22/downscaling/station_validation/201808_stationinfo.nc")#所有站点数据
station_info_1=xr.open_dataset("/stu01/chenss22/downscaling/station_validation/201801_stationinfo.nc")#所有站点数据
grid_nearest_info=xr.open_dataset("ERA5LAND_GD_coarse_8_interp.nc")#粗网格所有格点数据
grid_nearest_info_1=xr.open_dataset("ERA5LAND_GD_coarse_1_interp.nc")
grid_fine_info=xr.open_dataset("/stu01/chenss22/downscaling/station_validation/fine_data_after_downscaling/station_data/ERA5LAND_GD_fine_add_ws8_mergetime.nc")#细网格数据
grid_fine_info_1=xr.open_dataset("/stu01/chenss22/downscaling/station_validation/fine_data_after_downscaling/station_data/ERA5LAND_GD_fine_add_ws1_mergetime.nc")#细网格数据
A=["2018-08-01","2018-08-05","2018-08-09","2018-08-13","2018-08-17","2018-08-21","2018-08-29","2018-09-01"]
J=["2018-01-01","2018-01-05","2018-01-09","2018-01-13","2018-01-17","2018-01-21","2018-01-29","2018-02-01"]
stationID_origin=grid_nearest_station['stationID']
plot_timeslt(stationID_origin,grid_nearest_station,grid_nearest_info,grid_fine_info,station_info,"Aus",A)
grid_nearest_station=pd.read_excel("/stu01/chenss22/downscaling/station_validation/fine_data_after_downscaling/station_data/valitation_station.xlsx")#非缺失站点的信息
stationID_origin=grid_nearest_station['stationID']
plot_timeslt(stationID_origin,grid_nearest_station,grid_nearest_info_1,grid_fine_info_1,station_info_1,"Jan",J)
'''
j=0
index_min=0
index_max=736
fig1,axes1=plt.subplots(4,1)
for i in range(4):
  var_info_coarse=grid_nearest_info_1[vars[i]][station_low_index,index_min:index_max]
  var_info_fine=grid_nearest_info_1[vars[i]][station_low_index,index_min:index_max]
  station_info1=station_info[var_station[i]][station_low_index,index_min:index_max]
  station_info1.plot.line(x='Date', ax=axes1[i],
  color='black', linewidth=1.5, linestyle="solid", alpha=0.7, label='Observed')
  var_info_coarse.plot.line(x='Date', ax=axes1[i],
  linewidth=1.5, linestyle="solid", label=f'f{vars[i]} coarse', color=line_colors[j])
  var_info_fine.plot.line(x='Date', ax=axes1[i],
  linewidth=1.5, linestyle="solid", label=f'f{vars[i]} fine', color=line_colors[j+1])
  axes1[i].legend(loc='best', shadow=False, frameon=False, fontsize=30)   #color=line_colors
  j=j+1
plt.tight_layout()
plt.savefig("tqp_timeslt_Jan.png",dpi=100)
'''


    




