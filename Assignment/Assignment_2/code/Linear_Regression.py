import numpy as np
import matplotlib.pyplot as plt


# 计算loss
def liner_loss(w, b, data):
    x = data[:, 0]  # 代表的是第一列数据
    y = data[:, 1]  # 代表的是第二列数据
    # 损失函数:使用的是均方误差(MES)损失
    loss = np.sum((y - w * x - b) ** 2) / data.shape[0]
    # 返回loss
    return loss


# 计算梯度并更新参数
def liner_gradient(w, b, data, lr):

    # 数据集行数
    N = float(len(data))
    # 提取数据
    x = data[:, 0]
    y = data[:, 1]
    # 求梯度
    dw = np.sum(-(2 / N) * x * (y - w * x - b))
    db = np.sum(-(2 / N) * (y - w * x - b))
    # 更新参数
    w = w - (lr * dw)
    b = b - (lr * db)

    return w, b


# 每次迭代做梯度下降
def optimer(data, w, b, lr, epcoh):

    for i in range(epcoh):
        # 通过每次循环不断更新w,b的值
        w, b = liner_gradient(w, b, data, lr)
        # 每训练100次更新下loss值
        if i % 100 == 0:
            print('epoch {0}:loss={1}'.format(i, liner_loss(w, b, data)))

    return w, b

# 绘图
def plot_data(data, w, b):
    x = data[:, 0]
    y = data[:, 1]
    y_predict = w * x + b
    plt.plot(x, y, 'o')
    plt.plot(x, y_predict, 'k-')
    plt.show()

def liner_regression():
    # 加载数据
    data = np.loadtxt('../attachments/LinearRegression_data.csv', delimiter=',')
    # 显示原始数据的分布
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, 'o')
    plt.show()

    # 初始化参数
    lr = 0.01  # 学习率
    epoch = 1000  # 训练次数
    w = 0.0  # 权重
    b = 0.0  # 偏置
    # 输出各个参数初始值
    print('initial variables:\n initial_b = {0}\n intial_w = {1}\n loss of begin = {2} \n' \
          .format(b, w, liner_loss(w, b, data)))

    # 更新w和b
    w, b = optimer(data, w, b, lr, epoch)

    # 输出各个参数的最终值
    print('final formula parmaters:\n b = {1}\n w = {2}\n loss of end = {3} \n'.format(epoch, b, w,
                                                                                       liner_loss(w, b, data)))
    # 显示
    plot_data(data, w, b)


if __name__ == '__main__':
    liner_regression()