import gym
from gym import spaces
import cv2
from utils.pid import PID
import threading
import time
import os
import sys
from utils.deflist import *
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))  # 这代表添加当前路径的上三级目录
from xarm.wrapper import XArmAPI

class MREnv(gym.Env):
    def __init__(self):
        super(MREnv, self).__init__()

        # 初始化机械臂
        ip = ''
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        # self.arm.reset(wait=True)

        # 定义安全工作空间的范围
        self.safety_bounds = {
            'x': [0, 200],  # x 轴的最小值和最大值
            'y': [0, 300],  # y 轴的最小值和最大值
            'z': [0, 500],  # z 轴的最小值和最大值
            'roll': [-180, 180],
            'pitch': [-180, 180],
            'yaw': [-180, 180]
        }

        # 定义动作空间为6维位置 (x, y, z, rx, ry, rz)
        low = np.array([0, 0, 0, 0, 0, 0])  # 每个维度的最小值
        high = np.array([200, 300, 500, 180, 180, 180])  # 每个维度的最大值
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 定义观察空间为4维 (当前图像的2D位置 + 期望图像的2D位置)
        self.observation_space = spaces.Box(low=0, high=400, shape=(4,), dtype=np.float32)

        # 初始化观测和期望图像的位置
        self.current_image_position = np.array([0, 0])
        self.desired_image_position = np.array([200, 200])

        self.get_sensor_data = get_sensor_data

    def reset(self):
        # 将机器人移动到初始位置
        self.arm.set_position(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, speed=100, wait=True)

        # 重置当前和期望图像的位置
        self.current_image_position = self.get_sensor_data()
        self.desired_image_position = np.array([200, 200])

        # 返回初始观测
        observation = np.concatenate([self.current_image_position, self.desired_image_position])
        return observation

    def step(self, action):
        # 检查目标位置是否超出安全范围
        if not self.is_safe_position(action):
            print("安全限制触发，重置机械臂")
            return self.reset(), -10, True, {}  # 触发重置并给予负奖励

        # 执行动作
        self.arm.set_position(x=action[0], y=action[1], z=action[2], roll=action[3], pitch=action[4], yaw=action[5], speed=100, wait=True)

        # 检查机械臂状态，是否存在故障
        if self.is_faulty_state():
            print("机械臂故障，触发重置")
            return self.reset(), -20, True, {}  # 触发重置并给予更大的负奖励

        # 更新当前图像位置
        self.current_image_position = self.get_sensor_data()

        # 计算与期望图像位置的误差
        error = np.linalg.norm(self.current_image_position - self.desired_image_position)

        # 定义奖励函数，当误差在1以内时给予较高奖励
        if error < 1:
            reward = 10
            done = True
        elif error < 10:
            reward = 5
        else:
            reward = -error/100  # 或者使用其他奖励函数，如负的误差值
            done = False

        # 更新观测
        observation = np.concatenate([self.current_image_position, self.desired_image_position])

        return observation, reward, done, {}

    def render(self, mode='human'):
        # 打开摄像头并获取图像
        capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not capture.isOpened():
            print("无法打开摄像头")
            return

        while True:
            # 获取传感器数据 (图像中心点坐标)
            sensor_data = self.get_sensor_data()
            turn_center_x, turn_center_y = sensor_data

            # 读取当前摄像头帧
            retval, image = capture.read()

            if not retval:
                print("无法获取图像")
                break

            # 绘制中心点
            image_with_center = cv2.circle(image, (int(turn_center_x), int(turn_center_y)), 5, (0, 255, 0), -1)

            # 显示处理后的图像
            cv2.imshow("Camera View with Center", image_with_center)

            # 等待键盘输入，按下空格退出
            key = cv2.waitKey(200)  # 200毫秒刷新
            if key == 32:  # 按空格键退出
                break

        # 释放摄像头
        capture.release()
        cv2.destroyAllWindows()

    def close(self):
        pass

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

    def is_faulty_state(self):
        """
        检查机械臂状态是否故障
        """
        code, error = self.arm.get_err_warn_code()  # 获取机械臂的错误状态
        if code != 0:  # 如果存在错误码
            print(f"机械臂错误码: {code}, 错误信息: {error}")
            return True
        return False

def get_sensor_data():
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
    start = time.time()

    capture.isOpened()  # 笔记本内置摄像头被打开后
    imun = imun + 1
    now = time.time() - start

    # 原始图像数据
    retval, image = capture.read()  # 从摄像头中实时读取视频
    # 连接不稳定
    if image is None:
        image = cv2.imread("default.png")
        capture.release()  # 关闭摄像头
        capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)  # 打开摄像头0，外置1

    # 在窗口中显示读取到的视频
    cv2.imshow("origin_picture", image)

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
