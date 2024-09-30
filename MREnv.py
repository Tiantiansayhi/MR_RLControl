import gym
from gym import spaces
import cv2
from utils.pid import PID
import serial
import threading
import time
import os
import sys
from utils.deflist import *
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))  # 这代表添加当前路径的上三级目录
from xarm.wrapper import XArmAPI
import numpy as np

class MREnv(gym.Env):
    def __init__(self):
        super(MREnv, self).__init__()

        # 初始化机械臂
        ip = '192.168.1.199'
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        # 定义安全工作空间的范围
        self.safety_bounds = {
            'x': [-80, 0],  # x 轴的最小值和最大值
            'y': [-60, 30],  # y 轴的最小值和最大值
            'z': [-400, 100],  # z 轴的最小值和最大值
            'roll': [-180, 180],
            'pitch': [-180, 180],
            'yaw': [-180, 180]
        }

        # 定义动作空间为6维位置 (x, y, z, rx, ry, rz)
        a_low = np.array([-80, -60, -400, 0, 0, 0])  # 每个维度的最小值
        a_high = np.array([0, 30, 100, 180, 180, 180])  # 每个维度的最大值
        self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)

        # 定义观察空间为5维 (当前图像的2D位置 + 期望图像的2D位置+ imu A_z)
        o_low = np.array([0, 0, 0, 0, -50])  # 每个维度的最小值
        o_high = np.array([400, 400, 400, 180, 50])  # 每个维度的最大值
        self.observation_space = spaces.Box(low=o_low, high=o_high, shape=(5,), dtype=np.float32)

        # 初始化观测和期望图像的位置
        self.current_image_position = np.array([0, 0])
        self.desired_image_position = np.array([200, 200])
        self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 打开笔记本内置摄像头0，外置1（可能乱序）

        # 初始化IMU相关内容
        self.stop_thread = False  # 用于控制线程的运行

        # 1. 初始化IMU数据
        self.A_x = 0
        self.A_y = 0
        self.A_z = 0
        self.G_x = 0
        self.G_y = 0
        self.G_z = 0
        self.M_x = 0
        self.M_y = 0
        self.M_z = 0

        # 2. 创建线程锁
        self.imu_lock = threading.Lock()

        # 3. 初始化串口
        self.ser1 = serial.Serial('COM33', 115200)  # 实体串口：IMU

        # 4. 启动IMU数据采集线程,线程在主程序退出时自动结束，可以将线程设置为守护线程
        self.thread2 = threading.Thread(target=self.read_serial_0, daemon=True)
        self.thread2.start()

    def reset(self):
        # 将机器人移动到初始位置(需修改)
        self.arm.set_position(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, speed=100, wait=True)

        # 重置当前和期望图像的位置
        self.current_image_position = self.get_sensor_data()
        self.desired_image_position = np.array([200, 200])

        # 重置IMU观测
        self.A_x = 0
        self.A_y = 0
        self.A_z = 0
        self.G_x = 0
        self.G_y = 0
        self.G_z = 0
        self.M_x = 0
        self.M_y = 0
        self.M_z = 0

        # 返回初始观测
        observation = np.concatenate([self.current_image_position, self.desired_image_position, [self.A_z]])
        return observation

    def step(self, action):
        # 检查目标位置是否超出安全范围
        if not self.is_safe_position(action):
            print("安全限制触发，重置机械臂")
            return self.reset(), -10, True, {}  # 触发重置并给予负奖励

        # 执行动作
        self.arm.set_position(x=action[0], y=action[1], z=action[2], roll=action[3], pitch=action[4], yaw=action[5], speed=100, wait=True)

        # # 检查机械臂状态，是否存在故障
        # if self.is_faulty_state():
        #     print("机械臂故障，触发重置")
        #     return self.reset(), -20, True, {}  # 触发重置并给予更大的负奖励

        # 更新当前图像位置
        self.current_image_position = self.get_sensor_data()
        # 计算与期望图像位置的误差
        fig_error = np.linalg.norm(self.current_image_position - self.desired_image_position)

        # 使用锁读取IMU数据
        with self.imu_lock:
            imu_data = [self.A_x, self.A_y, self.A_z, self.G_x, self.G_y, self.G_z, self.M_x, self.M_y, self.M_z]

        # 图像奖励 + imu奖励 单步
        if imu_data[2] != 0:
            imu_reward = 110
        else:
            imu_reward = 0
        reward = -fig_error + imu_reward
        done = reward > 100

        # 更新观测
        observation = np.concatenate([self.current_image_position, self.desired_image_position, self.A_z])

        return observation, reward, done, {}

    def render(self, mode='human'):
        if not self.capture.isOpened():
            print("无法打开摄像头")
            return

        # 获取传感器数据 (图像中心点坐标)
        sensor_data = self.get_sensor_data()
        turn_center_x, turn_center_y = sensor_data

        # 读取当前摄像头帧
        retval, image = self.capture.read()

        if not retval:
            print("无法获取图像")
            return

        # 绘制中心点
        image_with_center = cv2.circle(image, (int(turn_center_x), int(turn_center_y)), 5, (0, 255, 0), -1)

        # 显示处理后的图像
        cv2.imshow("Camera View with Center", image_with_center)
        cv2.waitKey(1)

    def close(self):
        # 关闭摄像头
        if self.capture.isOpened():
            self.capture.release()
        # 关闭串口
        if self.ser1.is_open:
            self.ser1.close()
        # 停止IMU数据采集线程
        self.stop_thread = True
        if self.thread2.is_alive():
            self.thread2.join()

    def is_safe_position(self, action):
        """
        检查目标位置是否在安全工作空间范围内
        """
        x, y, z, roll, pitch, yaw = action
        return (self.safety_bounds['x'][0] <= x <= self.safety_bounds['x'][1] and
                self.safety_bounds['y'][0] <= y <= self.safety_bounds['y'][1] and
                self.safety_bounds['z'][0] <= z <= self.safety_bounds['z'][1] and
                self.safety_bounds['roll'][0] <= roll <= self.safety_bounds['roll'][1] and
                self.safety_bounds['pitch'][0] <= pitch <= self.safety_bounds['pitch'][1] and
                self.safety_bounds['yaw'][0] <= yaw <= self.safety_bounds['yaw'][1])

    # def is_faulty_state(self):
    #     """
    #     检查机械臂状态是否故障
    #     """
    #     code, error = self.arm.get_err_warn_code()  # 获取机械臂的错误状态
    #     if code != 0:  # 如果存在错误码
    #         print(f"机械臂错误码: {code}, 错误信息: {error}")
    #         return True
    #     return False

    def read_serial_0(self):
        Slong = 12  # 一个串口数据长度
        self.ser1.readline().decode('utf-8').rstrip() #预先读一行，丢弃

        while not self.stop_thread:
            try:
                read_ser1 = self.ser1.readline().decode('utf-8').rstrip()
                if read_ser1 != '':
                    line = read_ser1  # 读取的字符串
                    with self.imu_lock:  # 使用锁保护数据更新
                        self.A_x = float(line[6:13]) - 2000
                        self.A_y = float(line[(6 + 1 * Slong):(13 + 1 * Slong)]) - 2000
                        self.A_z = float(line[(6 + 2 * Slong):(13 + 2 * Slong)]) - 2000
                        self.G_x = float(line[(6 + 3 * Slong):(13 + 3 * Slong)]) - 2000
                        self.G_y = float(line[(6 + 4 * Slong):(13 + 4 * Slong)]) - 2000
                        self.G_z = float(line[(6 + 5 * Slong):(13 + 5 * Slong)]) - 2000
                        self.M_x = float(line[(6 + 6 * Slong):(13 + 6 * Slong)]) - 2000
                        self.M_y = float(line[(6 + 7 * Slong):(13 + 7 * Slong)]) - 2000
                        self.M_z = float(line[(6 + 8 * Slong):(13 + 8 * Slong)]) - 2000
            except (ValueError, IndexError) as e:
                print(f"串口数据解析错误：{e}")
                continue


    def get_sensor_data(self):
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
        start = time.time()

        self.capture.isOpened()  # 笔记本内置摄像头被打开后
        if not self.capture.isOpened():
            print("无法打开摄像头")
            return np.array([0, 0])
        imun = imun + 1
        now = time.time() - start

        # 原始图像数据
        retval, image = self.capture.read()  # 从摄像头中实时读取视频
        if not retval:
            print("无法获取图像")
            # 处理方式，例如返回默认值或抛出异常
            return np.array([0, 0])
        # 连接不稳定
        if image is None:
            image = cv2.imread("default.png")
            self.capture.release()  # 关闭摄像头
            self.capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # 打开摄像头0，外置1

        # 在窗口中显示读取到的视频
        # cv2.imshow("origin_picture", image)

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
        # cv2.imshow("down", G1)
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
        # cv2.imshow("can", fin)
        contours, hierarchy = cv2.findContours(fin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        v3 = G1.copy()
        res = cv2.drawContours(v3, contours, -1, (0, 0, 0), 3)

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
            # 排除最外层轮廓
            if G_threshold[i] > G_MAX and area > 2000.0:
                c_r = cnt
                G_MAX = G_threshold[i]

        # 在计算mu之前，检查G_MAX是否被更新，如果没有，说明未找到有效轮廓，直接返回默认值。
        if G_MAX == -9999:
            print("未找到有效的轮廓")
            return np.array([0, 0])
        mu = cv2.moments(c_r)
        # area1 = area
        try:
            mc = (mu['m10'] / (mu['m00']), mu['m01'] / (mu['m00']))
        except (ZeroDivisionError, KeyError):
            mc = (0, 0)

        ori_center_x = mc[0]
        ori_center_y = mc[1]
        # print(mc)
        v4 = cv2.circle(res, (int(mc[0]), int(mc[1])), 5, (220, 0, 0), -1)
        # cv2.imshow("point", v4)

        # 理论中心点
        turn_center_x0 = ((G1.shape[0]) / 2) - 0
        turn_center_y0 = ((G1.shape[1]) / 2) + 0
        # 实际中心点
        turn_center_x = mc[0]
        turn_center_y = mc[1]
        # 误差显示输出
        error1 = turn_center_x - turn_center_x0
        error2 = -(turn_center_y - turn_center_y0)
        return np.array([turn_center_x, turn_center_y])
