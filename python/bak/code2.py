import numpy as np
import matplotlib.pyplot as plt

# 动态权重的参数
def alpha(t, T=50, tau=10):
    """历史信念的动态权重。"""
    return 1 / (1 + np.exp((t - T / 2) / tau))

def beta(t, T=50, tau=10):
    """当前信号的动态权重。"""
    return 1 - alpha(t, T, tau)

def dynamic_kalman_gain(t, T=50):
    """动态卡尔曼增益以适应信号不确定性。"""
    return 1 / (1 + np.exp((t - T / 2) / 5))  # 较小的tau用于更快的调整

def kalman_filter(signal, z, K):
    """应用卡尔曼滤波器以抑制噪声。"""
    return signal + K * (z - signal)

# 模拟设置
T = 60  # 总时间步数
psi_signal = 0.7  # 真实后验概率（真实值）
initial_beliefs = [0.2, 0.5, 0.8]  # 不同的初始信念
noise_variance = 0.05  # 高斯噪声的减小方差

# 初始化信念数组
beliefs = {init: np.zeros(T) for init in initial_beliefs}
for init in initial_beliefs:
    beliefs[init][0] = init  # 设置初始信念

# 信念收敛模拟
np.random.seed(42)  # 为了可重复性
for init in initial_beliefs:
    for t in range(1, T):
        a = alpha(t)
        b = beta(t)
        noisy_signal = psi_signal + np.random.normal(0, np.sqrt(noise_variance))  # 含噪声信号
        K_t = dynamic_kalman_gain(t)  # 动态卡尔曼增益
        filtered_signal = kalman_filter(psi_signal, noisy_signal, K_t)  # 应用卡尔曼滤波器
        beliefs[init][t] = a * beliefs[init][t - 1] + b * filtered_signal

# 绘制信念收敛
plt.figure(figsize=(12, 6))
for init, belief in beliefs.items():
    plt.plot(belief, label=f"初始信念: {init}")
# plt.axhline(y=psi_signal, color="red", linestyle="--", label="真实后验概率")
plt.title("不同初始值下的信念收敛", fontsize=16)
plt.xlabel("时间步 $t$", fontsize=14)
plt.ylabel("信念 $\psi(t)$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# 噪声抑制鲁棒性比较
belief_no_filter = np.zeros(T)  # 无卡尔曼滤波器的信念
belief_with_filter = np.zeros(T)  # 有卡尔曼滤波器的信念
belief_no_filter[0] = belief_with_filter[0] = 0.5  # 共同的初始信念

# 信念更新模拟（有无卡尔曼滤波器）
for t in range(1, T):
    a = alpha(t)
    b = beta(t)
    noisy_signal = psi_signal + np.random.normal(0, np.sqrt(noise_variance))  # 含噪声信号
    belief_no_filter[t] = a * belief_no_filter[t - 1] + b * noisy_signal
    K_t = dynamic_kalman_gain(t)
    filtered_signal = kalman_filter(psi_signal, noisy_signal, K_t)
    belief_with_filter[t] = a * belief_with_filter[t - 1] + b * filtered_signal

# 计算方差和准确性
variance_no_filter = np.var(belief_no_filter)
variance_with_filter = np.var(belief_with_filter)
accuracy_no_filter = np.mean(np.abs(belief_no_filter - psi_signal) < 0.1)
accuracy_with_filter = np.mean(np.abs(belief_with_filter - psi_signal) < 0.1)

# 绘制噪声抑制结果
plt.figure(figsize=(12, 6))
plt.plot(belief_no_filter, label="无卡尔曼滤波器的信念", linestyle="-")
plt.plot(belief_with_filter, label="有卡尔曼滤波器的信念", linestyle="--")
# plt.axhline(y=psi_signal, color="red", linestyle=":", label="真实后验概率")
plt.title("噪声抑制对信念更新的影响", fontsize=16)
plt.xlabel("时间步 $t$", fontsize=14)
plt.ylabel("信念 $\psi(t)$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# 显示指标
print("信念更新的方差:")
print(f"无卡尔曼滤波器: {variance_no_filter}")
print(f"有卡尔曼滤波器: {variance_with_filter}\n")

print("噪声条件下的准确性:")
print(f"无卡尔曼滤波器: {accuracy_no_filter * 100:.2f}%")
print(f"有卡尔曼滤波器: {accuracy_with_filter * 100:.2f}%")
