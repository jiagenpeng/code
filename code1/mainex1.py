import PyQt5
import rasterio

from rasterio.plot import show,adjust_band
from mpl_toolkits.mplot3d import Axes3D

from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import pyqtSlot, Qt
import pyqtgraph as pg
import os, shutil, glob
from osgeo import gdal
import os
import sys

from PyQt5 import QtWidgets
import numpy as np
# import rasterio
# from rasterio import _shim,control,coords,crs,drivers,dtypes,enums,env,errors,features,fill,io
# from rasterio import mask,merge,path,plot,profiles,rpc,sample,session,tools,transform,vrt,warp,windows
import arcgis
from arcgis.gis import GIS
import pyqtgraph.opengl as gl
from visu import Ui_Form
import pylas
# from laspy.file import File
import matplotlib
import time
import matplotlib.pyplot as plt
import matplotlib.colors
from pyqtgraph.Qt import QtCore, QtGui
import pandas as pd
import operator
from functools import reduce


class HistMainWindow(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)  # 调用父类构造函数，创建窗体
        self.setupUi(self)  # 构造UI界面
        self.setWindowTitle("plot")
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.dotdata.clicked.connect(self.dot)  # 读取数据
        self.pushButton_2.clicked.connect(self.dtm)
        self.dsmdata.clicked.connect(self.dsm)

        self.DOT2.clicked.connect(self.drawdot)#三个按钮
        self.DTM2.clicked.connect(self.drawDTM)
        self.DSM2.clicked.connect(self.drawDSM)

        self.DOT3.clicked.connect(self.drawdot3)  # 三个按钮
        self.DTM3.clicked.connect(self.drawDTM3)
        self.DSM3.clicked.connect(self.drawDSM3)
        # self.pushButton.clicked.connect(lambda: self.restart())


    #  ==============自定义功能函数========================
    def dot(self):
        filePath = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件", "/")
        listn = []
        global con_x, con_y, con_z, con_x1, con_y1, con_z1
        con_x = []
        con_y = []
        con_z = []
        con_x1 = []
        con_y1 = []
        con_z1 = []

        fs = os.listdir(filePath)
        for f1 in fs:
            tmp_path = os.path.join(filePath, f1)
            if not os.path.isdir(tmp_path):
                listn.append(tmp_path)
        t = 0

        for i in listn:
            lazfile = i
            lazfile.encode('ascii', 'ignore').decode()
            data = pylas.read(lazfile)
            c = max(data.z)
            p = min(data.z)
            d = max(t, c)
            q = d
            o = min(q, p)
            if o != q:
                o = min(o, p)
            con_x.append(data.x)
            con_y.append(data.y)
            con_z.append(data.z)

        lazfile1 = listn[0]
        lazfile1.encode('ascii', 'ignore').decode()
        data = pylas.read(lazfile1)
        con_x1 = data.x
        con_y1 = data.y
        con_z1 = data.z
        for i in range(1, len(listn)):
            lazfile = listn[i]
            lazfile.encode('ascii', 'ignore').decode()
            data = pylas.read(lazfile)
            con_x1 = np.concatenate((con_x1, data.x), axis=None)
            a = con_x1
            con_y1 = np.concatenate((con_y1, data.y), axis=None)
            con_z1 = np.concatenate((con_z1, data.z), axis=None)

        global maxd, mind
        maxd = d
        mind = o


    def dtm(self):
        # filePath = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件", "/")
        # listn = []
        # global arrayq,s
        # arrayq=[]
        # s=[]
        # fs = os.listdir(filePath)
        # for f1 in fs:
        #     tmp_path = os.path.join(filePath, f1)
        #     if not os.path.isdir(tmp_path):
        #         listn.append(tmp_path)
        # arrayq = rasterio.open(listn[0])
        # c = arrayq.bounds
        # w = c[3] - c[1]
        # h = c[2] - c[0]
        # s.append(w * h / 1000000)  ##单位 km2
        # arrayq = arrayq.read(1)
        # for i in range(1,len(listn)):
        #     lazfiledtm = listn[i]
        #     array1 = rasterio.open(lazfiledtm)
        #     k=array1
        #     c = array1.bounds
        #     w = c[3] - c[1]
        #     h = c[2] - c[0]
        #     s.append(w * h / 1000000)##单位 km2
        #     arrayq = np.concatenate((arrayq, array1.read(1)), axis=None)
        #     k=arrayq
        # s=sum(s)
        # arrayq = arrayq.reshape(-1, 2500)
        filePath = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件", "/")

        global raster

        # fs = os.listdir(filePath)
        # for f1 in fs:
        #     tmp_path = os.path.join(filePath, f1)
        #     if not os.path.isdir(tmp_path):
        #         listn.append(tmp_path)

        out_file = "D:\\毕设\\code_end\\out_dtm"

        if os.path.exists(out_file):
            shutil.rmtree(out_file)
            os.mkdir(out_file)
        else:
            os.mkdir(out_file)
        data_list = glob.glob(filePath + "\\" + "*.tif")
        print(data_list)
        # for file in data_list:
        #     basename = os.path.basename(file)
        #     out_path = (out_file + "\\" + basename)

        try:

            # 读取其中一个栅格数据来确定镶嵌图像的一些属性
            o_ds = gdal.Open(data_list[0])

            # 投影
            Projection = o_ds.GetProjection()
            # 波段数据类型
            o_ds_array = o_ds.ReadAsArray()

            if 'int8' in o_ds_array.dtype.name:
                datatype = gdal.GDT_Byte
            elif 'int16' in o_ds_array.dtype.name:
                datatype = gdal.GDT_UInt16
            else:
                datatype = gdal.GDT_Float32

            # 像元大小
            transform = o_ds.GetGeoTransform()
            pixelWidth = transform[1]
            pixelHeight = transform[5]

            del o_ds

            minx_list = []
            maxX_list = []
            minY_list = []
            maxY_list = []

            # 对于每一个需要镶嵌的数据，读取它的角点坐标
            for data in data_list:
                # 读取数据
                ds = gdal.Open(data)
                rows = ds.RasterYSize
                cols = ds.RasterXSize

                # 获取角点坐标
                transform = ds.GetGeoTransform()
                minX = transform[0]
                maxY = transform[3]
                pixelWidth = transform[1]
                pixelHeight = transform[5]  # 注意pixelHeight是负值
                maxX = minX + (cols * pixelWidth)
                minY = maxY + (rows * pixelHeight)

                minx_list.append(minX)
                maxX_list.append(maxX)
                minY_list.append(minY)
                maxY_list.append(maxY)

                del ds

            # 获取输出图像坐标
            minX = min(minx_list)
            maxX = max(maxX_list)
            minY = min(minY_list)
            maxY = max(maxY_list)

            # 获取输出图像的行与列
            cols = int((maxX - minX) / pixelWidth)
            rows = int((maxY - minY) / abs(pixelHeight))  # 注意pixelHeight是负值

            # 计算每个图像的偏移值
            xOffset_list = []
            yOffset_list = []
            i = 0

            for data in data_list:
                xOffset = int((minx_list[i] - minX) / pixelWidth)
                yOffset = int((maxY_list[i] - maxY) / pixelHeight)
                xOffset_list.append(xOffset)
                yOffset_list.append(yOffset)
                i += 1

            # 创建一个输出图像
            driver = gdal.GetDriverByName("GTiff")
            dsOut = driver.Create(out_file + "\\" + "out.tif", cols, rows, 1, datatype)
            bandOut = dsOut.GetRasterBand(1)

            i = 0
            # 将原始图像写入新创建的图像
            for data in data_list:
                # 读取数据
                ds = gdal.Open(data)
                data_band = ds.GetRasterBand(1)
                data_rows = ds.RasterYSize
                data_cols = ds.RasterXSize

                data = data_band.ReadAsArray(0, 0, data_cols, data_rows)
                bandOut.WriteArray(data, xOffset_list[i], yOffset_list[i])

                del ds
                i += 1

            # 设置输出图像的几何信息和投影信息
            geotransform = [minX, pixelWidth, 0, maxY, 0, pixelHeight]
            dsOut.SetGeoTransform(geotransform)
            dsOut.SetProjection(Projection)

            del dsOut
        except:
            print(False)
        raster = rasterio.open(out_file + "\\" + "out.tif")
    def dsm(self):
        # filePath = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件", "r")
        # listn = []
        # global arraydsm, cellsizes
        # arraydsm = []
        # arraydsm = np.ndarray(arraydsm)
        # fs = os.listdir(filePath)
        # for f1 in fs:
        #     tmp_path = os.path.join(filePath, f1)
        #     if not os.path.isdir(tmp_path):
        #         listn.append(tmp_path)
        # arraydsm = np.loadtxt(listn[0], skiprows=6)
        # for i in range(1,len(listn)):
        #     lazfiledsm=listn[i]
        #     c=lazfiledsm
        #     lazfiledsm = np.loadtxt(lazfiledsm, skiprows=6)
        #     arraydsm = np.concatenate((arraydsm, lazfiledsm),axis=None)
        # arraydsm=arraydsm.reshape(-1,500)
        # framefile = pd.read_csv(c,encoding="gbk",engine='python',sep=' ',delimiter=None, index_col=False,header=None,skipinitialspace=True)
        # cellsizes = framefile.iat[4, 1]
        filePath = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件", "r")
        global pointx, pointy, Z, maxz, minz
        minX = []
        maxX = []
        minY = []
        maxY = []
        X = []
        Y = []
        Z = []

        data_list = glob.glob(filePath + "\\" + "*.asc")

        for i in data_list:
            file = open(i, 'r')
            ncols = file.readline().strip()
            cols = ''.join(list(filter(str.isdigit, ncols)))
            cols = int(cols)
            nrows = file.readline().strip()
            rows = ''.join(list(filter(str.isdigit, nrows)))
            rows = int(rows)
            xllcorner = file.readline().strip()
            minx = ''.join(list(filter(str.isdigit, xllcorner)))
            minx = int(minx)
            yllcorner = file.readline().strip()
            miny = ''.join(list(filter(str.isdigit, yllcorner)))
            miny = int(miny)
            cellsize = file.readline().strip()
            cellsize = ''.join(list(filter(str.isdigit, cellsize)))
            cellsize = int(cellsize)

            maxx = minx + (cols * cellsize)
            maxy = miny + (cellsize * rows)
            minX.append(minx)
            maxX.append(maxx)
            minY.append(miny)
            maxY.append(maxy)
            arraydsm = np.loadtxt(i, skiprows=6)
            for row in range(rows):
                for col in range(cols):
                    if arraydsm[row][col] != -9999:
                        X.append(minx + col * cellsize)
                        Y.append(miny - (rows - row - 1) * cellsize)
                        Z.append(arraydsm[row][col])
        minx_new = min(minX)
        maxx_new = max(maxX)
        miny_new = min(minY)
        maxy_new = max(maxY)
        pointx = []
        pointy = []
        for i in X:
            a = i - minx_new
            pointx.append(a)
        for i in Y:
            a = i - miny_new
            pointy.append(a)
        maxz = max(Z)
        minz = min(Z)
        print(minX)
        print(len(pointy), len(pointx))
        print(minY)


    def drawdot(self):  # 绘制2D-点图
        # 显示数据
        self.graphicsView.clear()
        self.p = self.graphicsView.addPlot()
        for i in range(0,len(con_x)):
            self.x = con_x[i]
            self.y = con_y[i]
            self.z = con_z[i]
            self.p.plot(self.x, self.y)
        self.p.show()
        self.label_max.setText(str(int(maxd))+'m')
        self.label_min.setText(str(int(mind))+'m')
        self.label_cell.setText('--')
        self.label_s.setText('--')

    def drawdot3(self):  # 绘制3D-点图
        # 显示数据
        self.graphicsView.clear()
        self.p = self.graphicsView.addPlot()
        self.x = con_x1
        self.y = con_y1
        self.z = con_z1
        cmap = plt.cm.get_cmap("rainbow")
        norm = matplotlib.colors.Normalize(vmin=min(self.z), vmax=max(self.z))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z, s=0.0005 * np.array(120 - self.z) ** 2, color=cmap(norm(self.z)), marker=".")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm)
        ax.grid(False)
        plt.show()
        self.label_max.setText(str(int(maxd)) + 'm')
        self.label_min.setText(str(int(mind)) + 'm')
        self.label_cell.setText('--')
        self.label_s.setText('--')


    def drawDTM(self):  # 绘制2D-DTM
        # 显示数据
        self.graphicsView.clear()
        # self.p = self.graphicsView.addPlot()  # 点一次按钮，生成一个新图
        app = pg.mkQApp("GLSurfacePlot Example")
        w = gl.GLViewWidget()

        show((raster, 1))

    def drawDTM3(self):  # 绘制3D-DTM
        # 显示数据
        self.graphicsView.clear()
        # self.p = self.graphicsView.addPlot()  # 点一次按钮，生成一个新图
        nrows = raster.height
        ncols = raster.width
        Xmin = raster.bounds.left
        Xmax = raster.bounds.right
        Ymin = raster.bounds.bottom
        Ymax = raster.bounds.top
        x = np.linspace(Xmin, Xmax, ncols)
        y = np.linspace(Ymin, Ymax, nrows)
        X, Y = np.meshgrid(x, y)
        Z = raster.read(1)
        region = np.s_[10:400, 10:400]
        X, Y, Z = X[region], Y[region], Z[region]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow")
        plt.show()


    def drawDSM(self):  # 绘制2D-DSM
        # 显示数据
        self.graphicsView.clear()
        self.p = self.graphicsView.addPlot()
        # for i in range(0, len(pointx)):
        self.x = pointx
        self.y = pointy
        self.z = Z
        self.p.plot(self.x, self.y)
        self.p.show()
        self.label_max.setText(str(int(maxz)) + 'm')
        self.label_min.setText(str(int(minz)) + 'm')
        self.label_cell.setText('--')
        self.label_s.setText('--')


    def drawDSM3(self):  # 绘制2D-DSM
        # 显示数据
        self.graphicsView.clear()
        self.p = self.graphicsView.addPlot()
        self.x = pointx
        self.y = pointy
        self.z = Z
        cmap = plt.cm.get_cmap("rainbow")
        norm = matplotlib.colors.Normalize(vmin=min(self.z), vmax=max(self.z))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z, color=cmap(norm(self.z)), marker=".")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm)
        ax.grid(False)
        plt.show()
        self.label_max.setText(str(int(maxz)) + 'm')
        self.label_min.setText(str(int(minz)) + 'm')
        self.label_cell.setText('--')
        self.label_s.setText('--')


if __name__ == "__main__":

    app = QApplication(sys.argv)  # 创建GUI应用程序
    form = HistMainWindow()  # 创建窗体
    form.show()
    sys.exit(app.exec_())







