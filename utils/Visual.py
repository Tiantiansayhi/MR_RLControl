"""
作者:f
日期:2024年07月3日
"""

# 导入opencv包
import cv2
import numpy as np
from pid import PID
import threading

import cv2
import time
import os
import sys
from deflist import *

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))  # 这代表添加当前路径的上三级目录

# pid参数
workNum = 1
if workNum == 0:
    pid1 = PID(p=0, i=0, d=0, imax=90)
    pid2 = PID(p=0, i=0, d=0, imax=90)
else:
    pid1 = PID(p=20, i=0, d=0.008, imax=90)
    pid2 = PID(p=20, i=0, d=0.008, imax=90)

# 打开摄像头
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 打开笔记本内置摄像头0，外置1（可能乱序）
# 端口号，根据自己实际情况输入，可以在设备管理器查看
# 虚拟串口调试方案
# ser1 = serial.Serial('COM33', 115200)  # 实体串口：IMU
# ser3 = serial.Serial('COM30', 115200)  # 实体串口：磁控
# ser1 = serial.Serial('COM2', 9600)  # 虚拟串口
thread2 = threading.Thread(name='t2', target=read_serial_0, args=())
# thread2.start()  # 启动线程2
time.sleep(2)
MPU = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # MPU原始数据
# 轮廓default定义
c_r = 0
c_r2 = 0
mu = 0
mu2 = 0
ori_center_x = 0
ori_center_y = 0
ori_center_x2 = 0
ori_center_y2 = 0
area = 0
area1 = 0
area2 = 0
mpu_x = 0
mpu_y = 0
mpu_z = 0
# 允许误差像素
error_range = 6

hsv_low = np.array([86, 21, 13])
hsv_high = np.array([174, 117, 78])

imun = 0  # 运行次数
#
# line = '100.00,100.00'
# HE = '2000,2000,2000,2000'
# data_list = [[] for _ in range(100000)]  # 预分配list长度 程序运行速度有待提升
# data_oneline = []
start = time.time()

while capture.isOpened():  # 笔记本内置摄像头被打开后
    # 读取mpu数据
    # MPU[0] = A_x
    # MPU[1] = A_y
    # MPU[2] = A_z
    # MPU[3] = G_x
    # MPU[4] = G_y
    # MPU[5] = G_z
    # MPU[6] = M_x
    # MPU[7] = M_y
    # MPU[8] = M_z
    # mpu_X = mpu_x
    # mpu_Y = mpu_y
    # mpu_Z = mpu_z
    imun = imun + 1
    now = time.time() - start

    # 原始图像数据
    retval, image = capture.read()  # 从摄像头中实时读取视频
    # 连接不稳定
    if image is None:
        image = cv2.imread("default.png")
        capture.release()  # 关闭摄像头
        capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # 打开摄像头0，外置1

    # 静态图片为视频源
    # cv2.imwrite("copy.png", image) # 保存图片
    # image = cv2.imread("copy.png")  # 读取图片
    # image_test = cv2.imread("getPhoto/get2.jpg")  # 读取图片
    # 在窗口中显示读取到的视频
    cv2.imshow("origin_picture", image)
    # print('Ax:%.3f Ay:%.3f Az:%.3f Gx:%.3f Gy:%.3f Gz:%.3f Mx:%.3f My:%.3f Mz:%.3f' % (MPU[0], MPU[1], MPU[2], MPU[3], MPU[4], MPU[5], MPU[6], MPU[7], MPU[8]))

    # 1畸变处理
    # data_dist = np.load('data1.npz')
    # mtx = data_dist['mtx']
    # dist = data_dist['dist']
    # h, w = image.shape[:2]
    # newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数
    # 进行畸变处理
    # image_dist = cv2.undistort(image, mtx, dist, None, newCameraMtx)
    # 跳过处理
    image_dist = image
    # image_dist = image_test  # 静态图片
    cut = 0  # 黑边裁剪
    image_dist = image_dist[0 + cut:400 - cut, 0 + cut:400 - cut]
    new_size = (400, 400)  # 调用resize函数进行缩放
    image_dist = cv2.resize(image_dist, new_size)
    # cv2.imshow("dist_picture", image_dist)

    # 2直方图均衡化
    # rgb转hsv
    image_hsv = cv2.cvtColor(image_dist, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    # 对亮度通道执行直方图均衡化
    equalized_v = cv2.equalizeHist(v)
    # 合并均衡化后的亮度通道与饱和度、色调通道
    equalized_hsv_image = cv2.merge([h, s, equalized_v])
    # 将均衡化后的图像转换回RGB颜色空间
    equalized_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)
    # 红色通道图像的提取
    # red_channel = equalized_image[:, :, 2]
    red_channel = image_dist[:, :, 2]
    # 中值滤波
    blur = cv2.medianBlur(red_channel, 5)
    # cv2.imshow("hsv", blur)
    # 降采样
    G0 = blur
    G1 = cv2.pyrDown(G0)
    # L0 = G0 - cv2.pyrUp(G1)
    # RO = L0 + cv2.pyrUp(G1)  # 通过拉普拉斯图像复原的原始图像
    cv2.imshow("down", G1)
    # 最大灰度
    Lmin, Lmax, minloc_1, maxloc_1 = cv2.minMaxLoc(G1)
    mean_1 = cv2.mean(G1)
    Lmean = mean_1[0]
    hist = cv2.calcHist([G1], [0], None, [256], [0, 256])
    n_p = ((G1.shape[0]) / 2) * ((G1.shape[1]) / 2)
    K_threshold = [0 for i in range(256)]
    G_threshold = [0 for i in range(256)]
    seta_T = [0 for i in range(256)]

    # 对图像进行Sobel边缘检测
    x = cv2.Sobel(G1, cv2.CV_64F, 1, 0)  # 垂直方向梯度
    y = cv2.Sobel(G1, cv2.CV_64F, 0, 1)  # 水平方向梯度

    absX = cv2.convertScaleAbs(x)  # 转回原来的uint8格式，否则图像无法显示。
    absY = cv2.convertScaleAbs(y)  # 转回原来的uint8格式，否则图像无法显示

    Gv = absX
    Gh = absY
    GmeanM = (Gv ** 2 + Gh ** 2) ** 0.5
    Gmean = cv2.mean(GmeanM)[0]
    # print(Gmean)
    P_Ac0 = hist[0]
    T_Ac0 = 0

    K_MAX = -9999
    for i in range(1, int(Lmax)):
        if hist[i] == 0:
            hist[i] = 1
        seta_T[i] = (Lmax / (i * Gmean)) * hist[i] * ((i - Lmean) ** 2) / n_p
    for i in range(1, int(Lmax)):
        # 累积像素数量和像素灰度合
        P_Ac0 = hist[i] + P_Ac0
        T_Ac0 = hist[i] * i + T_Ac0
        Pc0 = P_Ac0 / n_p
        Pc1 = 1 - Pc0
        T_c0 = T_Ac0 / P_Ac0
        T_c1 = (Lmean * n_p - T_Ac0) / (n_p - P_Ac0)
        seta_B = Pc0 * Pc1 * (T_c1 - T_c0)
        K_threshold[i] = ((Lmax - i) / Gmean) * (seta_B / seta_T[i])
        if K_threshold[i] > K_MAX:
            K_MAX = K_threshold[i]
            T_adap = i - 50
    # print(T_adap)

    # 画轮廓，
    _, frame2 = cv2.threshold(G1, T_adap, 255, cv2.THRESH_BINARY)  # 二值化
    fin = cv2.Canny(frame2, 100, 255)  # Canny边缘检测  40 110
    cv2.imshow("can", fin)
    contours, hierarchy = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    v3 = G1.copy()
    res = cv2.drawContours(v3, contours, -1, (0, 0, 0), 3)
    # cv2.imshow("drawContours", res)
    # 边缘检测
    # frame0 = cv2.GaussianBlur(image_dist, (5, 5), 0)  # 高斯模糊
    # frame1 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)  # 灰度化
    # _, frame2 = cv2.threshold(frame1, 70, 255, cv2.THRESH_BINARY)  # 二值化
    # # frame2 = cv2.GaussianBlur(image_dist, (5, 5), 0)  # 高斯模糊
    # # dst = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)  # BGR转HSV
    #
    # # frame1 = cv2.inRange(dst, hsv_low, hsv_high)
    #
    # fin = cv2.Canny(frame2, 40, 110)  # Canny边缘检测  40 110

    # cv2.imshow("black", frame2)
    # 轮廓抖动、断续问题
    # cv2.imshow("Canny", fin)
    # 轮廓识别
    #
    # contours, hierarchy = cv2.findContours(G0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # fin-g0
    # v2 = image_dist.copy()

    # res = cv2.drawContours(v2, contours, -1, (0, 0, 255), 2)
    # cv2.imshow("drawContours", res)
    # 特征提取
    # 找到最大圆心位置,第二大圆心位置，进行圆心连线，求连线角度和水平角度偏转角

    # drawing = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)  # 创建画布
    # drawing = v3 # 画在原图副本上
    # 识别二个圆 轮廓抖动、断续问题
    # 轮廓选择
    G_MAX = -9999
    for i in range(1, int(len(contours))):

        area = cv2.contourArea(contours[i])
        mask = np.zeros(G1.shape, np.uint8)
        cnt = contours[i]
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(G1, mask=mask)

        mean_val = cv2.mean(G1, mask=mask)

        try:
            G_threshold[i] = (area ** 0.5) / (mean_val[0] * min_val * min_val)
        except ZeroDivisionError:
            G_threshold[i] = 0
        print(i)
        print(G_threshold[i])
        # 排除最外层轮廓
        if G_threshold[i] > G_MAX and area > 2000.0:
            c_r = cnt
            G_MAX = G_threshold[i]

    # c_r = contours[1]
    mu = cv2.moments(c_r)
    # area1 = area
    try:
        mc = (mu['m10'] / (mu['m00']), mu['m01'] / (mu['m00']))
    except ZeroDivisionError:
        mc = 0, 0

    ori_center_x = mc[0]
    ori_center_y = mc[1]
    # print(mc)
    v4 = cv2.circle(res, (int(mc[0]), int(mc[1])), 5, (220, 0, 0), -1)
    cv2.imshow("point", v4)
    # feature = (float(mc[0]), float(mc[1]), float(area1) * pow(10, -2))
    # cv2.drawContours(drawing, c_r, -1, (255, 0, 0,), 2)
    # cv2.drawContours(v3, c_r, -1, (0, 255, 0,), 2)

    # for i in range(len(contours)):
    #
    #     area = cv2.contourArea(contours[i])
    #     # 找出大圆
    #     # 7000- 8000//2000-1500     8500- 6500//2500-1000
    #     if 8000 > area > 6500:
    #         c_r = contours[i]
    #         mu = cv2.moments(c_r)
    #         area1 = area
    #         mc = (mu['m10'] / (mu['m00']), mu['m01'] / (mu['m00']))
    #         ori_center_x = mc[0]
    #         ori_center_y = mc[1]
    #         cv2.circle(drawing, (int(mc[0]), int(mc[1])), 4, (0, 220, 0), -1)
    #         feature = (float(mc[0]), float(mc[1]), float(area1) * pow(10, -2))
    #         # cv2.drawContours(drawing, c_r, -1, (255, 0, 0,), 2)
    #         cv2.drawContours(v3, c_r, -1, (0, 255, 0,), 2)
    #
    #     # 找出小圆
    #     if 2500 > area > 1500:
    #         c_r2 = contours[i]
    #         mu2 = cv2.moments(c_r2)
    #         area2 = area
    #         mc2 = (mu2['m10'] / (mu2['m00']), mu2['m01'] / (mu2['m00']))
    #         ori_center_x2 = mc2[0]
    #         ori_center_y2 = mc2[1]
    #         cv2.circle(drawing, (int(mc2[0]), int(mc2[1])), 4, (0, 220, 0), -1)
    #         feature2 = (float(mc2[0]), float(mc2[1]), float(area2) * pow(10, -2))
    #         # cv2.drawContours(drawing, c_r2, -1, (155, 130, 130,), 2)
    #         cv2.drawContours(v3, c_r2, -1, (0, 255, 0,), 2)

    # # 标定畸变(可以在读取图片时就完成此步)
    # plane_center_x = plane(ori_center_x)
    # plane_center_y = plane(ori_center_y)
    # plane_center_x2 = plane(ori_center_x2)
    # plane_center_y2 = plane(ori_center_y2)
    #
    # # 偏转角调控(弧度制) - imu数据
    # # 小圆朝向x1
    # # alpha = math.atan2((plane_center_y2 - plane_center_y), (plane_center_x2 - plane_center_x))ZZ  # 偏转角度
    # # alpha = mpu_Z
    # alpha = 0
    # M = cv2.getRotationMatrix2D((200, 200), alpha, 1)
    # # M = cv2.getRotationMatrix2D((200, 200), math.degrees(alpha), 1)
    # turn_center_x = 200 + math.cos(-alpha) * (plane_center_x - 200) - math.sin(-alpha) * (plane_center_y - 200)
    # turn_center_y = 200 + math.sin(-alpha) * (plane_center_x - 200) + math.cos(-alpha) * (plane_center_y - 200)
    # turn_center_x2 = 200 + math.cos(-alpha) * (plane_center_x2 - 200) - math.sin(-alpha) * (plane_center_y2 - 200)
    # turn_center_y2 = 200 + math.sin(-alpha) * (plane_center_x2 - 200) + math.cos(-alpha) * (plane_center_y2 - 200)
    # # turn_center_x2 =
    # # turn_center_y2 =
    # v4 = cv2.warpAffine(v3, M, (400, 400))
    # cv2.circle(v4, (int(turn_center_x), int(turn_center_y)), 4, (0, 0, 220), -1)
    # # cv2.drawContours(v4, c_r, -1, (255, 0, 0,), 2)
    # cv2.circle(v4, (int(turn_center_x2), int(turn_center_y2)), 4, (0, 0, 220), -1)

    # pid控制 偏转显示与图像一致

    # turn_center_x0 = (image.shape[0])/2  # 理论中心200,200,总宽400，400
    # turn_center_y0 = (image.shape[1])/2

    # 理论中心点
    turn_center_x0 = ((G1.shape[0]) / 2) - 0
    turn_center_y0 = ((G1.shape[1]) / 2) + 0
    # 实际中心点
    turn_center_x = mc[0]
    turn_center_y = mc[1]
    # 稳定点中心x-31.632137082902148,y-9.756924498615064,几何中心x-33.88834802459891,y17.40770460507062,
    # cv2.circle(v4, (int(turn_center_x0), int(turn_center_y0)), 4, (0, 220, 220), -1)
    # cv2.imshow("roll", v4)
    # 误差显示输出
    error1 = turn_center_x - turn_center_x0
    error2 = -(turn_center_y - turn_center_y0)

    # 驱动量算法

    # 驱动机械臂
    # # 图像与实际匹配
    # output1 = pid1.get_pid(-error1)
    # output2 = pid2.get_pid(-error2)
    #
    # # 传输控制值 001 002 PWM满幅是3600 限制在3500
    # output_limit = 1250
    # limit(output1, -output_limit, output_limit)
    #
    # # 磁控稳态线圈输出值
    # # output1 = -850
    # # output2 = 800
    # # 将传输数据转化为int
    # data01 = round(output1, 3)
    # data02 = round(output2, 3)
    #
    # plus_value = 1000
    # mlu_value = 1  # 坐标偏移方向和实际胶囊偏移反向
    # data001 = round(mlu_value * data01 + plus_value)
    # data002 = round(mlu_value * data02 + plus_value)

    # 传输控制值 导轨控制与偏移相反与图像相同

    # if error1 > error_range:
    #     data1x = '+'
    # elif error1 < -error_range:
    #     data1x = '-'
    # else:
    #     data1x = '0'
    #
    # if error2 > error_range:
    #     data1y = '+'
    # elif error2 < -error_range:
    #     data1y = '-'
    # else:
    #     data1y = '0'
    #
    # data1 = data1x + data1y + '/'
    # print('send:' + data1)
    # 传输控制值 原视觉伺服
    # data1 = 'x' + str(data001) + ",y" + str(data002) + ',\r\n'
    # data_pc = 'x' + str(data01) + ",y" + str(data02) + ',\r\n'
    # # 显示坐标
    # data_pc = '偏差：x' + str(error1) + ",y" + str(error2) + ',\r\n' + '输出：x' + str(output1) + ",y" + str(
    #     output2) + ',\r\n' + '面积：大' + str(area1) + ",小" + str(area2) + ',\r\n '
    # # 显示面积
    # data_pc = 'x' + str(area1) + ",y" + str(area2) + ',\r\n'
    # ser1.write(data1.encode('utf-8'))
    # ser2.write(data1.encode('utf-8'))
    # ser1.write(b'++\n')
    # ser2.write(b'++\n')

    # print(data_pc)
    # print(' X:%.3f Y :%.3f Z:%.3f' % (mpu_X, mpu_Y, mpu_Z))

    # 读取串口信息
    # read_ser1 = ser1.read_all().decode('utf-8').rstrip()
    # if read_ser1 != '':
    #     line = read_ser1
    # print(line)
    # line = ser1.read_all()
    # parts = line.split(",", 2)
    # read_ser3 = ser3.readline().decode('utf-8').rstrip()
    # if read_ser3 != '':
    #     HE = read_ser3
    # # data_oneline = [imun, now, parts[0], parts[1], error1, error2, HEparts[3], HEparts[1], HEparts[2], HEparts[0]]
    # data_oneline = [imun, now, line, area1, error1, error2, HE, 0, 0, 0]
    # # data_list.append(data_oneline)
    # data_list[imun-1] = data_oneline

    # print('圆心坐标为：x=', center_x, 'y=', center_y, '轮廓面积为：', area)
    # cv2.imshow('Contours2', drawing)
    # 位置计算
    # 圆心在像素的位置*比例关系到现实位置
    # 展示图像处理始末
    # imgStack = np.vstack((image_dist, v4))
    # cv2.imshow("Capture and Analysis", imgStack)
    # 展示图像处理步骤
    # imgStack2 = np.vstack((frame2, fin))
    # imgStack3 = np.hstack((imgStack,imgStack2 ))
    # cv2.imshow("Analysis process", imgStack2)

    # control_x = center_x
    # control_y = center_y

    key = cv2.waitKey(200)  # 窗口的图像刷新时间为x毫秒 200-5hz
    if key == 32:  # 如果按下空格键
        break

capture.release()  # 关闭笔记本内置摄像头
cv2.destroyAllWindows()  # 销毁显示摄像头视频的窗口
# ser1.close()
# ser2.close()
# ser3.close()
# # 表格数据统计
# # print(data_list)
# workbook = xlwt.Workbook()
# sheet = workbook.add_sheet('Sheet1', cell_overwrite_ok=True)
# daytime = datetime.now()
# sheet.write(0, 0, daytime.now().strftime('%F %T'))
# sheet.write(0, 1, '运行时间')
# sheet.write(0, 2, '导轨x坐标')
# sheet.write(0, 3, '导轨y坐标')
# sheet.write(0, 4, '图像误差x坐标')
# sheet.write(0, 5, '图像误差y坐标')
# sheet.write(0, 6, '图像面积z误差')
# sheet.write(0, 7, 'x1霍尔读数')
# sheet.write(0, 8, 'x2霍尔读数')
# sheet.write(0, 9, 'y1霍尔读数')
# sheet.write(0, 10, 'y2霍尔读数')
# sheet.write(0, 11, 'x轴读数差')
# sheet.write(0, 12, 'y轴读数差')
# for im in range(0, imun):
#     # print(im)
#     imp = im+1
#     # print(imun)
#     sheet.write(imp, 0, data_list[im][0])
#     sheet.write(imp, 1, data_list[im][1])
#     parts = data_list[im][2].split(",", 2)
#     # sheet.write(imp, 2, data_list[im][2])
#     sheet.write(imp, 2, parts[0])
#     sheet.write(imp, 3, parts[1])
#     sheet.write(imp, 4, data_list[im][4])
#     sheet.write(imp, 5, data_list[im][5])
#     sheet.write(imp, 6, data_list[im][3])
#     # y2 x2 y1 x1 ——  0 1 2 3
#     HEparts = data_list[im][6].split(",")
#     sheet.write(imp, 7, HEparts[3])
#     sheet.write(imp, 8, HEparts[1])
#     sheet.write(imp, 9, HEparts[2])
#     sheet.write(imp, 10, HEparts[0])
#     sheet.write(imp, 11, float(HEparts[3])-float(HEparts[1]))
#     sheet.write(imp, 12, float(HEparts[2])-float(HEparts[0]))
#     # 面积特征
#     # sheet.write(imp, 6, data_list[im][6])
#     # sheet.write(imp, 7, data_list[im][7])
#     # sheet.write(imp, 8, data_list[im][8])
#     # sheet.write(imp, 9, data_list[im][9])
# print("写入完毕")
# workbook.save('data_xlwt/don3.xls')
# # 五个稳定数据 wen1
# # 五个偏移表格 jin1
# # 五个运动表格 don1
