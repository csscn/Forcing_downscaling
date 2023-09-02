import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader as shpreader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.image import imread
import shapefile
import shapely.geometry as sgeom
from osgeo import gdal
import tifffile as tf   
import xarray as xr 
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

shp_path = '/stu01/chenss22/data_downscaling/MERITDEM_parameter/MERITDEM_ncfile/gd_shp/'
# extent = [70, 140, 0, 60]  #图2范围
extent = [109, 118, 20, 26]  #图3范围
# --- 加载中国矢量图信息
Chinese_land_province = shp_path +"Export_Output_2.shp"

# --- 加载全球高分辨率地形
tif_path = "/stu01/chenss22/data_downscaling/MERITDEM_parameter/MERITDEM_ncfile/"
# --- 加载站点信息
df=pd.read_excel("gd_station.xlsx")
# --- 加载地形nc文件
ncfil=xr.open_dataset(tif_path+"MERITDEM_height.nc")


def create_map(lon,lat):
    # 创建坐标系
    prj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 10), dpi=350)
    axes = fig.subplots(1, 1, subplot_kw={'projection': prj})
    axes.set_extent(extent, crs=prj)


    #mask GuangDong province
    """
    x=np.linspace(start=0,stop=10799,num=10800)
    y=np.linspace(start=0,stop=7199,num=7200)
    X,Y=np.meshgrid(x,y)
    mask = np.zeros(X.shape, dtype=bool)
    with shapefile.Reader(Chinese_land_province)as reader:
        guangdong=sgeom.shape(reader.shape(0))
    for index in np.ndindex(X.shape):
        point=sgeom.Point(X[index],Y[index])
    if guangdong.contains(point):
        mask[index]=True
    else:
        mask[index]=False
    #ar=imread(tif_path + 'guangdong_clip.tif')
    #ax.add_feature(Chinese_land_territory, linewidth=1)
    #x=np.linspace(0,10799)
    #y=np.linspace(0,7199)
    #X,Y=np.meshgrid(x,y)
    #cset=plt.contourf(X,Y,ar,8,cmap="rainbow")
    """
    dataset = gdal.Open(tif_path + 'MERITDEM_height_float_clip.tif')
    arr = dataset.ReadAsArray()
    arr[np.where(arr<0)]=np.nan #arr==-340282306073709652508363335590014353408

    LON,LAT=np.meshgrid(lon,lat)
    ax_fig=axes.contourf(LON,LAT,arr,transform=prj,cmap="gray")
    #divider = make_axes_locatable(axes)
    cax = inset_axes(axes, width="3%", height="20%", loc='lower right')#bbox_to_anchor=(0.8, 0., 1, 1))#, borderpad=0)
    cbar=plt.colorbar(ax_fig,cax=cax)
    cbar.ax.yaxis.label.set_fontsize(6)
    cbar.ax.set_title("m",fontsize=6)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=6)
    #cbar_min, cbar_max=0,2400
    #cbar.mappable.set_clim(cbar_min, cbar_max)
    #ax.imshow(tf.imread(tif_path + 'MERITDEM_height_float_clip.tif'),
     #   origin='upper', transform=prj,
      #  extent=[109, 118, 20, 29]
    #)
    gl = axes.gridlines(crs=prj, draw_labels=True, linewidth=1.2, color='k',
    alpha=0.5, linestyle='--')
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1]+1, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3]+1, 1))
        # 绘制广东省界
    Chinese_land_territory = shpreader(Chinese_land_province).geometries()
    Chinese_land_territory = cfeat.ShapelyFeature(Chinese_land_territory,
                                              prj, edgecolor='g',
                                              facecolor='none')
    axes.add_feature(Chinese_land_territory, linewidth=1)

    return axes
    # --绘制散点图
df['lon'] = df['lon'].astype(np.float64)
df['lat'] = df['lat'].astype(np.float64)
ax = create_map(ncfil['lon'],ncfil['lat'])
#df['lon'] = df['lon'].astype(np.float64)
#df['lat'] = df['lat'].astype(np.float64)
ax.scatter(
  df['lon'].values,
  df['lat'].values,
  marker='o',
  s=10 ,
  color ="red"
)

# --添加浮标名称
for i, j, k in list(zip(df['lon'].values, df['lat'].values, df['station'].values)):
  ax.text(i - 0.2, j + 0.05, k, fontsize=6,color="white")

# --添加标题&设置字号
title = f'distribution of station around GuangDong'
ax.set_title(title, fontsize=18)
#plt.colorbar(cset)
# 绘制网格点

plt.savefig('guangdong_station.png')
