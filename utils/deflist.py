import numpy as np
def gaussian_distribution_generator(var, num):
    return np.random.normal(loc=0.0, scale=var, size=num)


def rgb_to_hsv(r, g, b):
    # 将RGB值转换为0-1范围内的百分比
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    # 计算最大值和最小值
    max_value = max(r, g, b)
    min_value = min(r, g, b)

    # 计算色相（Hue）
    if max_value == min_value:
        hue = 0
    elif max_value == r:
        hue = ((g - b) / (max_value - min_value)) % 6
    elif max_value == g:
        hue = ((b - r) / (max_value - min_value)) + 2
    elif max_value == b:
        hue = ((r - g) / (max_value - min_value)) + 4

    hue *= 60

    # 计算饱和度（Saturation）
    if max_value == 0:
        saturation = 0
    else:
        saturation = 1 - (min_value / max_value)

    # 计算明度（Value）
    value = max_value

    return (hue, saturation, value)


# # 示例：将RGB颜色(128, 64, 192)转换为HSV颜色
# r, g, b = 128, 64, 192
# hsv = rgb_to_hsv(r, g, b)
# print(hsv)

def iterate_x(x_in):
    ret = np.zeros(len(x_in))
    w = np.array(gaussian_distribution_generator(0.01, 4))
    for i in range(len(x_in)):
        ret[i] = x_in[i]
    return ret


def iterate_z(x_in, delta_q):
    ret = np.zeros(2)
    # TODO:H = np.eye(6)_deltaq
    H = np.zeros((2, 4))

    for i in range(2):
        H[i, 2 * i:2 * i + 2] = delta_q
        ret[i] = np.dot(H[i], x_in)
    return ret


def check_delta(theta, step):
    max_theta = np.array([500, 100, 600, 180, 40, 180])
    min_theta = np.array([300, -100, 310, -180, -40, -180])

    max_step = [0.2, 0.2, 7, 0, 0, 0]
    for i in range(0, 6):
        # limit the delta
        while abs(step[i]) > max_step[i]:
            step[i] = 0.5 * step[i]

        theta[i] += step[i]
        # limit the max_theta
        theta[i] = max_theta[i] if theta[i] > max_theta[i] else theta[i]
        # limit the min_theta
        theta[i] = min_theta[i] if theta[i] < min_theta[i] else theta[i]


"""
Just for test example
"""


# 平面畸变函数
def plane(ori_center):
    plane_center = ori_center
    return plane_center


# imu姿态角读取函数
def read_serial_1():
    global mpu_x
    global mpu_y
    global mpu_z
    ser1.readline().decode('utf-8').rstrip()
    while 1:
        read_ser1 = ser1.readline().decode('utf-8').rstrip()
        if read_ser1 != '':
            line = read_ser1
            mpu_x = float(line[5:11]) - 360
            mpu_y = float(line[15:21]) - 360
            mpu_z = float(line[25:31]) - 360
            # print(line + '\n')
            # print(' X:%.3f Y:%.3f Z:%.3f' %(mpu_x, mpu_y, mpu_z))


# imu原始数据读取函数
def read_serial_0():
    global A_x
    global A_y
    global A_z
    global G_x
    global G_y
    global G_z
    global M_x
    global M_y
    global M_z
    Slong = 12  # 一个串口数据长度
    ser1.readline().decode('utf-8').rstrip()
    while 1:
        read_ser1 = ser1.readline().decode('utf-8').rstrip()
        if read_ser1 != '':
            line = read_ser1  # 读取的字符串
            # 减去串口stm32加上内的数值，使符号一致
            A_x = float(line[6:13]) - 2000
            A_y = float(line[(6 + 1 * Slong):(13 + 1 * Slong)]) - 2000
            A_z = float(line[(6 + 2 * Slong):(13 + 2 * Slong)]) - 2000
            G_x = float(line[(6 + 3 * Slong):(13 + 3 * Slong)]) - 2000
            G_y = float(line[(6 + 4 * Slong):(13 + 4 * Slong)]) - 2000
            G_z = float(line[(6 + 5 * Slong):(13 + 5 * Slong)]) - 2000
            M_x = float(line[(6 + 6 * Slong):(13 + 6 * Slong)]) - 2000
            M_y = float(line[(6 + 7 * Slong):(13 + 7 * Slong)]) - 2000
            M_z = float(line[(6 + 8 * Slong):(13 + 8 * Slong)]) - 2000
            # print(line + '\n')
            # print('Ax:%.3f Ay:%.3f Az:%.3f Gx:%.3f Gy:%.3f Gz:%.3f Mx:%.3f My:%.3f Mz:%.3f' % ( A_x, A_y, A_z, G_x, G_y, G_z, M_x, M_y, M_z))


# 最值限制函数
def limit(value, lower, upper):
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value
