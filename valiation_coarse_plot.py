import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
def scatter_plot(i,j,axes,x,y,z,before,u,u_num0,cbar_min,cbar_max):
  units=["(k)","(%)","(hPa)","(k)"]
  if i==1:
   ax=axes[i][j].scatter(x,y,c=z*u_num0[1],s=2,alpha=0.4,cmap='Spectral') 
   #u="e-3"
   #cbar_min=0.0
   #cbar_max=1.2
   axes[i][j].set_xticks([40,50,60,70,80,90,100])
   axes[i][j].set_yticks([40,50,60,70,80,90,100])
  else:
   ax=axes[i][j].scatter(x,y,c=z*u_num0[0],s=2,alpha=0.4,cmap='Spectral')
   axes[i][j].set_xticks([292,294,296,298,300,302])
   axes[i][j].set_yticks([292,294,296,298,300,302])
  # u="e-1"
  # cbar_min=0.0
  # cbar_max=1.4
  axes[i][j].plot((0,1),(0,1),transform=axes[i][j].transAxes,c='k')
  axes[i][j].set_title(var_station[i]+f" {before} downscaling",fontsize=8)
  cbar=fig.colorbar(ax,ax=axes[i][j])
  #cbar.set_label("e-2")
  cbar.ax.yaxis.label.set_fontsize(8)
  cbar.ax.tick_params(labelsize=8)
  cbar.ax.set_title(u,fontsize=8)
  cbar.mappable.set_clim(cbar_min, cbar_max)
  axes[i][j].set_xlim(np.nanmin(data_station),np.nanmax(data_station))
  axes[i][j].set_ylim(np.nanmin(data_station),np.nanmax(data_station))
  axes[i][j].set_xlabel(f"{units[i]}",fontsize=8)
  axes[i][j].set_ylabel(f"{units[i]}",fontsize=8)
  axes[i][j].tick_params(axis='both',labelsize=8)
  return axes

def probability_density(data_station,data_grid):
 # Calculate the point density
  x=data_station[0,:]
  ix=np.where(np.logical_not(np.isnan(x)))
  y=data_grid[0,:]
  iy=np.where(np.logical_not(np.isnan(y)))
  result = np.intersect1d(ix, iy)
  print(x.reshape)
  x=x[result]
  y=y[result]
  xy = np.vstack([x,y])
  print(type(xy))
  print(len(xy))
  z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
  idx = z.argsort()
  x, y, z =np.array(x)[idx], np.array(y)[idx], np.array(z)[idx]
  return x,y,z

def rh(data_grid,grid_nearest_info):
 #马古列斯函数用到的常数
 epsi=.62198 # Ratio of molecular weight of water and dry air
 abs0=-273.15
 AW1, BW1, CW1 = 611.21, 17.502, 240.97  
 AW2, BW2, CW2 = 611.21, 17.368, 238.88
 AI,  BI,  CI  = 611.15, 22.452, 272.55
 m=data_grid/(1-data_grid) #混合比m=q/(1-q)
 Ta=grid_nearest_info["t2m"][index1,:]#0:736
 p=grid_nearest_info["sp"][index1,:]#0:736
 A=np.ones_like(Ta)*AW1
 B=np.ones_like(Ta)*BW1
 C=np.ones_like(Ta)*CW1
 A[Ta+abs0>5],B[Ta+abs0>5],C[Ta+abs0>5]=AW2, BW2, CW2
 A[Ta+abs0<-5],B[Ta+abs0<-5],C[Ta+abs0<-5]=AI,  BI,  CI  
 es=A*np.exp(B*(Ta+abs0)/(Ta+abs0+C)) #马古列斯公式计算饱和水汽压
 ms=epsi*es/(p-es)
 data_grid=m/ms*100 #计算最邻近格点相对湿度
 return data_grid
def r_rmse(index1,data_grid,data_grid_fine,data_station):
  data_grid=np.array(data_grid)
  data_grid_fine=np.array(data_grid_fine)
  data_station=np.array(data_station)
  num=len(index1)*744
  data_grid=data_grid.reshape(1,num)
  print(data_grid_fine.shape)
  data_grid_fine=data_grid_fine.reshape(1,num)
  data_station=data_station.reshape(1,num)
  #计算所有站点所有时间点的r^2,rmse，8月为67*744，1月为67*736
  num1=data_grid.shape
  mse=np.nansum((data_grid[0,:]-data_station[0,:])**2)/(num1[1]-np.isnan(data_grid).sum())
  mse_fine=np.nansum((data_grid_fine[0,:]-data_station[0,:])**2)/(num1[1]-np.isnan(data_grid_fine).sum())
  print("ws max value")
  print(np.nansum(data_grid)/(num1[1]-np.isnan(data_grid).sum()))
  print(np.nanmin(data_grid_fine)/(num1[1]-np.isnan(data_grid_fine).sum()))
  print(np.nanmin(data_station)/(num1[1]-np.isnan(data_station).sum()))
  rmse=np.sqrt(mse)
  rmse_fine=np.sqrt(mse_fine)
  r2=1-mse/np.nanvar(data_grid[0,:])
  r2_fine=1-mse_fine/np.nanvar(data_grid_fine[0,:])

  return rmse,rmse_fine,r2,r2_fine

##主程序##
grid_nearest_station=pd.read_excel("/stu01/chenss22/downscaling/station_validation/fine_data_after_downscaling/station_data/valitation_station.xlsx")#非缺失站点的信息
station_info=xr.open_dataset("/stu01/chenss22/downscaling/station_validation/201808_stationinfo.nc")#所有站点数据
grid_nearest_info=xr.open_dataset("ERA5LAND_GD_coarse_8_interp.nc")#粗网格所有格点数据
#细网格所有数据
grid_fine_info=xr.open_dataset("/stu01/chenss22/downscaling/station_validation/fine_data_after_downscaling/station_data/ERA5LAND_GD_fine_add_ws8_mergetime.nc")
grid_nearest_station["elevation_diff_coarse"]=np.abs(grid_nearest_station["elevation_diff_coarse"])
grid_nearest_station.sort_values(by="elevation_diff_coarse",inplace=True,ascending=True)#按照站点和格点的绝对海拔差进行排序
print(grid_nearest_station["elevation_diff_coarse"])
elevation_diff_fra=grid_nearest_station["elevation_diff_fra"]
index_col=grid_nearest_station["index"]#index_col表示在原站点序列中的位置
print(index_col)
n=len(elevation_diff_fra)
quantile=[1.0]
vars=["d2m","Q","sp","t2m"]
var_station=["td","rh","ps","t"]
units=["(k)","(%)","(hPa)","(k)"]
#space=[5,5,5,5]
rr2=np.zeros(2)
rrmse=np.zeros(2)
rr2_fine=np.zeros(2)
rrmse_fine=np.zeros(2)
index_before=0
kk=0
#画图参数设定
j=[0,1]
before=["before","after"]
u=["e-1","e-3"]
u_num0=[10,1000]
cbar_min=[0,0]
cbar_max=[1.4,1.2]
k=1
index=int(np.round(n*1)) #分位点计算
index1=index_col.iloc[index_before:index]
"""
for q in quantile:
 index=int(np.round(n*q)) #分位点计算
 index1=index_col.iloc[index_before:index]
 print(index-index_before)
 index_before=index
 fig,axes=plt.subplots(2,2)
 for i in range(2):  
  #grid_nearest_info1=grid_nearest_info[vars[i]][index_col,:]
  data_grid=grid_nearest_info[vars[i]][index1,:] #0:736
  data_grid_fine=grid_fine_info[vars[i]][index1,:]#0:736
  print(data_grid_fine.shape)
  data_station=station_info[var_station[i]][index1,:]
  if vars[i]=="sp":
   data_grid=data_grid/100.0
   data_grid_fine=data_grid_fine/100.0
  elif vars[i]=="Q":#由比湿计算相对湿度
   data_grid=rh(data_grid,grid_nearest_info)
   #细网格计算相对湿度
   data_grid_fine=rh(data_grid_fine,grid_fine_info)
  else:
   data_grid=data_grid
   data_grid_fine=data_grid_fine
  print(type(data_grid))
  data_grid=np.array(data_grid)
  data_grid_fine=np.array(data_grid_fine)
  data_station=np.array(data_station)
  num=len(index1)*744
  data_grid=data_grid.reshape(1,num)
  print(data_grid_fine.shape)
  data_grid_fine=data_grid_fine.reshape(1,num)
  data_station=data_station.reshape(1,num)
  #计算所有站点所有时间点的r^2,rmse，8月为67*744，1月为67*736
  num1=data_grid.shape
  mse=np.nansum((data_grid[0,:]-data_station[0,:])**2)/(num1[1]-np.isnan(data_grid).sum())
  mse_fine=np.nansum((data_grid_fine[0,:]-data_station[0,:])**2)/(num1[1]-np.isnan(data_grid_fine).sum())
  print(mse)
  rmse=np.sqrt(mse)
  rmse_fine=np.sqrt(mse_fine)
  r2=1-mse/np.nanvar(data_grid[0,:])
  r2_fine=1-mse_fine/np.nanvar(data_grid_fine[0,:])
  print(np.nanvar(data_grid[0,:]))
  rr2[i]=r2
  rr2_fine[i]=r2_fine
  rrmse[i]=rmse
  rrmse_fine[i]=rmse_fine
  #计算粗网格的概率密度并绘制散点图
  x,y,z=probability_density(data_station,data_grid)
  axes=scatter_plot(i,j[0],axes,x,y,z,before[0],u[i],u_num0,cbar_min[i],cbar_max[i])
  #细网格概率密度计算并绘制散点图
  x,y,z=probability_density(data_station,data_grid_fine)
  axes=scatter_plot(i,j[1],axes,x,y,z,before[1],u[i],u_num0,cbar_min[i],cbar_max[i])
 
#在图片相应位置打印出r^2和rmse
 axes[0][0].text(299,293,"$\mathregular{R^2}$="+f"{'%.2f'%rr2[0]}",fontsize=6)
 axes[0][1].text(299,293,"$\mathregular{R^2}$="+f"{'%.2f'%rr2_fine[0]}",fontsize=6)
 axes[1][0].text(82,45,"$\mathregular{R^2}$="+f"{'%.2f'%rr2[1]}",fontsize=6)
 axes[1][1].text(82,45,"$\mathregular{R^2}$="+f"{'%.2f'%rr2_fine[1]}",fontsize=6)
 axes[0][0].text(299,291,f"RMSE={'%.2f'%rrmse[0]}",fontsize=6)
 axes[0][1].text(299,291,f"RMSE={'%.2f'%rrmse_fine[0]}",fontsize=6)
 axes[1][0].text(82,42,f"RMSE={'%.2f'%rrmse[1]}",fontsize=6)
 axes[1][1].text(82,42,f"RMSE={'%.2f'%rrmse_fine[1]}",fontsize=6)
 plt.tight_layout()
 plt.savefig(f"td_rh_interp_8.png",dpi=300)
"""
#####风速的散点图
u10_coarse=grid_nearest_info["u10"][index1,:]
v10_coarse=grid_nearest_info["v10"][index1,:]
ws_fine=grid_fine_info["ws"][index1,:]
u10_sta=station_info["u"][index1,:]
v10_sta=station_info["v"][index1,:]
ws_coarse=np.sqrt(u10_coarse.values**2+v10_coarse.values**2)
ws_sta=np.sqrt(u10_sta.values**2+v10_sta.values**2)
#计算所有站点所有时间点的r^2,rmse，8月为67*744，1月为67*736
rmse,rmse_fine,r2,r2_fine=r_rmse(index1,ws_coarse,ws_fine,ws_sta)
print("#######results:")
print(rmse,rmse_fine,r2,r2_fine)
#计算粗网格的概率密度并绘制散点图
x,y,z=probability_density(ws_sta,ws_coarse)
fig1,ax=plt.subplots(1,2,figsize=[30,15])
ax1=ax[0].scatter(x,y,c=z,s=2,alpha=0.4,cmap='Spectral') 
cbar=fig1.colorbar(ax1,ax=ax[0])
ax[0].plot((0,1),(0,1),transform=ax[0].transAxes,c='k')
ax[0].set_xticks([0,2,4,6,8,10])
ax[0].set_yticks([0,2,4,6,8,10])
x,y,z=probability_density(ws_sta,ws_fine)
ax2=ax[1].scatter(x,y,c=z,s=2,alpha=0.4,cmap='Spectral')
cbar=fig1.colorbar(ax2,ax=ax[1]) 
ax[1].plot((0,1),(0,1),transform=ax[1].transAxes,c='k')
ax[1].set_xticks([0,2,4,6,8,10])
ax[1].set_yticks([0,2,4,6,8,10])
plt.tight_layout()
plt.savefig(f"ws.png",dpi=300)




 











