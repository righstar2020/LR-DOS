import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
# import seaborn as sns
# import scienceplots
# # # 设置Seaborn的主题为IEEE风格
# # sns.set_theme(style="whitegrid", palette="muted", font="Times New Roman")

# # 科学风格 + 明亮主题 + 中文支持
# plt.style.use(['science', 'bright', 'no-latex', 'cjk-sc-font'])
# # 设置字体为 Times New Roman
# font_path = './front_path/Times_New_Roman.ttf'
# font_manager.fontManager.addfont(font_path)
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.sans-serif'] = 'Times New Roman'



# 定义邻居关系（用于信念更新）
def generate_neighbors(N):
    neighbors = {i: [i - 1, i + 1] for i in range(N)}  # 每个节点的邻居（前后相邻节点）
    for i in neighbors:
        neighbors[i] = [n for n in neighbors[i] if 0 <= n < N]  # 排除越界的邻居
    return neighbors



def generate_X(signal):
    """
        生成所有节点的信号特征向量
        signal: 信号类型(s1:正常,s2:蜜罐)
        internal:
             - X_N:信号向量维度(属性的个数)
             - X_M:信号取值范围标签(0~9)
             - 标签单数为正常属性,双数为蜜罐属性
    """
    X_N = 10
    X_M = 10
    X = np.zeros(X_N)
    selected_indices = np.random.choice(X_N, 6, replace=False)
    if signal == "s1":
        for i in range(X_N):
            if i in selected_indices:
                X[i] = np.random.choice(range(1, X_M, 2))
            else:
                X[i] = np.random.randint(0,X_M)
    else:
        for i in range(X_N):
            if i in selected_indices:
                X[i] = np.random.choice(range(0,X_M,2))
            else:
                X[i] = np.random.randint(0,X_M)
    print(f"new X: {X}")
    return X



def caculate_attacker_observation(action_signal_matrix,theta,signal):
    """
        计算攻击者观测概率(条件概率)
        action_signal_matrix: 事件观察矩阵2*2
        theta: 攻击者行为(0:攻击,1:不攻击)
        signal: 防御者信号(0:正常,1:蜜罐)
        internal:
            - [0,0]:攻击者攻击,正常信号
            - [0,1]:攻击者攻击,蜜罐信号
            - [1,0]:攻击者撤退,正常信号
            - [1,1]:攻击者撤退,蜜罐信号
    """
    observation_theta = 0
    if theta == 0:
        if signal == 0:
            if sum(action_signal_matrix[0]) != 0:
                observation_theta = action_signal_matrix[0][0] / sum(action_signal_matrix[0])
            print(f"攻击者攻击,正常信号的概率: {observation_theta}")
        else:
            if sum(action_signal_matrix[0]) != 0:
                observation_theta = action_signal_matrix[0][1] / sum(action_signal_matrix[0])
            print(f"攻击者攻击,蜜罐信号的概率: {observation_theta}")
    else:
        if signal == 0:
            if sum(action_signal_matrix[1]) != 0:
                observation_theta = action_signal_matrix[1][0] / sum(action_signal_matrix[1])
            print(f"攻击者撤退,正常信号的概率: {observation_theta}")
        else:
            if sum(action_signal_matrix[1]) != 0:
                observation_theta = action_signal_matrix[1][1] / sum(action_signal_matrix[1])
            print(f"攻击者撤退,蜜罐信号的概率: {observation_theta}")
    return observation_theta


def calculate_EFD(X_list,normal_signal_nodes):
    """
        计算曝露虚假程度(EFD)
        X_list: 所有节点的信号特征向量列表
        normal_signal_nodes: 发送正常信号节点的索引列表
    """
    if len(X_list) == 0:
        return 0
    n = len(X_list)
    efd = 0
    #计算所有蜜罐节点与正常节点的属性向量距离
    honeypot_X_list = [X_list[i] for i in range(n) if i not in normal_signal_nodes]
    normal_X_list = [X_list[i] for i in range(n) if i in normal_signal_nodes]
    print(f"honeypot_X_list: {honeypot_X_list}")
    print(f"normal_X_list: {normal_X_list}")
    for honeypot_X in honeypot_X_list:
        for normal_X in normal_X_list:
            dist = np.linalg.norm(np.array(honeypot_X) - np.array(normal_X))
            efd += dist
    efd /= max(1,len(honeypot_X_list) * len(normal_X_list))
    print(f"new EFD: {efd}")
    return efd

def calculate_HTD(X_list):
    """
        计算隐藏真实程度(HTD)
        X_list: 所有节点的信号特征向量列表
        internal:
            - X_count_matrix: 属性向量计数矩阵
            - X_count_matrix[i][j]: 属性i的j类型出现的次数
    """
    if len(X_list) == 0:
        return 0
    htd = 0
    X_count_matrix = np.zeros((10, 10))
    for X in X_list:
        for i in range(len(X)):
            if 0 <= X[i] < 10:  # 确保 X[i] 在有效范围内
                X_count_matrix[i][int(X[i])] += 1  # 确保 X[i] 是整数
    X_count_matrix = X_count_matrix / max(1, np.sum(X_count_matrix))
    for i in range(10):
        for j in range(10):
            if X_count_matrix[i][j] > 0:  # 避免 log(0)
                htd += X_count_matrix[i][j] * np.log(X_count_matrix[i][j])
    htd = -htd

    print(f"new HTD: {htd}")
    return htd

def run_game_simulation(T=100,N=20):
    """
        运行信号博弈模拟
        param:
            T: 时间步数
            N: 节点数量
    """
    #----------------------------------------------------1.初始化-----------------------------------------------------------------
    # 初始化参数
    Ra_initial = 4  # 攻击者初始奖励
    Ca1_initial, Ca2_initial, Ca3 = 2, 5, 10  # 攻击者的初始成本(攻击合法系统,攻击伪蜜罐,攻击高交互蜜罐)
    Co = 1.5  # 攻击者探测的成本
    Rds = 6  # 防御者伪装信号的奖励
    CH = 5  # 防御者布置高交互蜜罐的成本
    Rinf = 5  # 防御者通过影响攻击者信念获得的奖励
    Rp = 5  # 攻击者避免被蜜罐捕获的奖励
    CDs = 1.5  # 防御者部署伪装的成本
    lambda_a = 0.4 # 攻击者奖励衰减因子
    lambda_2 = 0.4  # 攻击者成本衰减因子
    Rdec_initial = 8  # 防御者初始欺骗奖励
    lambda_d = 0.2  # 防御者奖励衰减因子

    # 时间步设置
    T = T  # 总时间步数
    N = N   # 节点数量（攻击目标或防御节点）

    # 初始信念值
    initial_belief = 0.5  # 攻击者对每个节点的初始信念
    beliefs = np.full((N,), initial_belief)  # 初始化所有节点的信念

    # 信念更新权重参数
    δ = 0.3
    T_half = T / 2  # 用于动态调整权重的中点

    # 初始化效用和结果
    attacker_utilities_t = []  # 每个时间步攻击者的总效用
    defender_utilities_t = []  # 每个时间步防御者的总效用
    EFD_t = []                 # 曝露虚假程度的历史记录
    HTD_t = []                 # 隐藏真实程度的历史记录

    # 初始化攻击者成功率
    success_rates = []  # 每个时间步攻击者成功率

    noise_scale_attacker = 0.01  # 攻击者效用噪声标准差
    noise_scale_defender = 0.01  # 防御者效用噪声标准差
    neighbors = generate_neighbors(N)
    

    # 初始化防御节点类型（真实系统 θ1 或蜜罐 θ2）
    defense_types = np.random.choice([0, 1], size=N, p=[0.5, 0.5])  # 随机初始化节点类型
    num_detected = 0  # 被蜜罐捕获的攻击次数
    num_probes = 0    # 攻击者探测次数
    #----------------------------------------------------2.统计数据-----------------------------------------------------------------
    #攻击者行为与信号事件计数attack:0,retreat:1,s1:0,s2:1
    action_signal_matrix = [[0,0],[0,0]]
    belief_s1_history = []        # 保存信念(综合信念,信号s1)的历史记录
    belief_s2_history = []        # 保存信念(综合信念,信号s2)的历史记录
    belief_normal_history_avg = []   # 保存信念(综合信念,节点类型正常)的历史平均记录
    belief_honeypot_history_avg = []   # 保存信念(综合信念,节点类型蜜罐)的历史平均记录
    signal_s1_count_history = []   # 保存信号s1的计数历史记录
    signal_s2_count_history = []   # 保存信号s2的计数历史记录
    #期望效用统计
    attacker_utility_s1_attack_history = []
    attacker_utility_s1_retreat_history = []
    attacker_utility_s2_attack_history = []
    attacker_utility_s2_retreat_history = []
    defender_utility_s1_history = []
    defender_utility_s2_history = []
    defender_utility_normal_history = []
    defender_utility_honeypot_history = []
    defender_utility_normal_s1_history = []
    defender_utility_honeypot_s1_history = []
    defender_utility_normal_s2_history = []
    defender_utility_honeypot_s2_history = []
    #----------------------------------------------------3.开始模拟-----------------------------------------------------------------
    # 主循环：逐时间步模拟交互
    for t in range(1, T + 1):
        # 动态调整信念更新权重
        alpha = 1 / (1 + np.exp(-(t - T_half) / (T / 10)))  # 自身信念的权重(时间步越大,权重越小)
        beta = 1 - alpha  # 来自伪装信号的权重
        gamma = δ * alpha * (1 - t / T)  # 邻居信念的权重

        # 初始化时间步的总效用
        attacker_utility_t = 0  # 当前时间步攻击者的总效用
        defender_utility_t = 0  # 当前时间步防御者的总效用
        new_beliefs = np.zeros_like(beliefs)  # 用于存储更新后的信念
        belief_update = 0
        #-------------------------------------------统计数据-----------------------------------------------------------------
        efd = 0  # 曝露虚假程度累加器
        htd = 0  # 隐藏真实程度累加器
        success_count = 0  # 攻击者成功攻击的次数
        # 节点属性向量集
        X_list = []
        normal_signal_nodes = []
        # 攻击者信念(不同信号下)
        psi_t_s1_avg = 0
        psi_t_s2_avg = 0
        # 针对不同节点类型信念累加
        psi_normal_avg = 0
        psi_honeypot_avg = 0
        # 防御选择信号的计数
        signal_s1_count = 0
        signal_s2_count = 0
        # 攻击者的期望效用(不同信号下)
        EU_A_s1_attack_t = 0
        EU_A_s1_retreat_t = 0
        EU_A_s2_attack_t = 0
        EU_A_s2_retreat_t = 0
        # 防御者的期望效用(不同信号下)
        EU_D_s1_t = 0
        EU_D_s2_t = 0
        # 防御者的期望效用(不同节点类型下)
        EU_D_normal_t = 0
        EU_D_honeypot_t = 0
        # 防御者的期望效用(不同节点下,不同信号)
        EU_D_normal_s1_t = 0
        EU_D_honeypot_s1_t = 0
        EU_D_normal_s2_t = 0
        EU_D_honeypot_s2_t = 0
        #-------------------------------------------------------------------------------------------------------------------
        print(f"\n时间步 {t}:")
        print("--------------------------------------------------------------------------------------------------------------")

        # 遍历所有节点
        #--------------------------------------------节点迭代 start--------------------------------------------
        for i in range(N):
            if t == 1:
                # 时间步 1，防御者随机选择信号（无攻击者的行动历史）
                signal = "s1" if np.random.rand() < 0.5 else "s2"
                psi_t_s1 = initial_belief  # 信念未更新，使用初始值
                psi_t_s2 = initial_belief
                # 初始化攻击者参数
                Ra_t = Ra_initial
                Ca1_t = Ca1_initial
                Ca2_t = Ca2_initial #攻击高交互蜜罐成本
                belief_update = initial_belief
            else:
                # 计算攻击者观测概率(根据攻击者历史行为计算)
                observation_theta1_s1 = caculate_attacker_observation(action_signal_matrix,0,0) #攻击,正常
                observation_theta1_s2 = caculate_attacker_observation(action_signal_matrix,0,1) #攻击,蜜罐
                observation_theta2_s1 = caculate_attacker_observation(action_signal_matrix,1,0) #不攻击,正常
                observation_theta2_s2 = caculate_attacker_observation(action_signal_matrix,1,1) #不攻击,蜜罐

                # 信念更新（原始信念）
                # 后验信念1-发出s1,攻击者攻击信念
                phi_theta1_s1 = beliefs[i] * observation_theta1_s1 / (beliefs[i] * observation_theta1_s1 + 
                                                                (1 - beliefs[i]) * observation_theta2_s1)
                phi_theta2_s1 = 1 - phi_theta1_s1
                # 后验信念2-发出s2,攻击者攻击信念
                phi_theta1_s2 = beliefs[i] * observation_theta1_s2 / (beliefs[i] * observation_theta1_s2 + 
                                                                (1 - beliefs[i]) * observation_theta2_s2)
                phi_theta2_s2 = 1 - phi_theta1_s2

                z_actual_s1 = phi_theta1_s1+np.random.rand()*0.1 #增加随机噪声(模拟观测值)
                z_actual_s2 = phi_theta1_s2+np.random.rand()*0.1 #增加随机噪声(模拟观测值)
                K_t = 0.8 / t  # 卡尔曼增益随时间增加而减少
                phi_prime_theta1_s1 = phi_theta1_s1 + K_t * (z_actual_s1 - phi_theta1_s1)  # 使用卡尔曼滤波更新
                phi_prime_theta1_s2 = phi_theta1_s2 + K_t * (z_actual_s2 - phi_theta1_s2)

                # 邻居综合信念的加权平均
                neighbor_beliefs = [beliefs[n] for n in neighbors[i]]
                neighbor_weights = np.array([1 / (abs(i - n) + 1) for n in neighbors[i]])
                neighbor_weights /= neighbor_weights.sum()
                psi_neighbor = np.dot(neighbor_weights, neighbor_beliefs)

                # 计算综合信念(历史综合信念+滤波信念+邻居信念)
                psi_t_s1 = alpha *beliefs[i] + beta * phi_prime_theta1_s1   + gamma * psi_neighbor
                psi_t_s2 = alpha *beliefs[i] + beta * phi_prime_theta1_s2   + gamma * psi_neighbor
                # 信念越大攻击者攻击的概率就越大(这个是攻击者历史信念，根据之前的信息更新)
                psi_t_s1 = max(0, min(1, psi_t_s1))  # 在发出s1信号(正常)时，攻击者攻击的概率
                psi_t_s2 = max(0, min(1, psi_t_s2))  # 在发出s2信号(蜜罐)时，攻击者攻击的概率

                # 计算防御者的期望效用(攻击者根据历史产生了信念，这种条件下防御者不同信号下的效用)
                if defense_types[i] == 0:  # 防御者节点类型 θ1(正常主机)
                    Ra_t = max(Ra_initial * np.exp(-lambda_a * np.log(t + 1)), 0)
                    Ca1_t = max(Ca1_initial * (1 - lambda_2 * min(num_probes, t / 2)), 0) #攻击合法系统成本随攻击者探测次数和时间步增加而增加
                    Ca2_t = max(Ca2_initial / (1 + lambda_2 * min(num_probes, t / 2)), 0.1) #攻击伪蜜罐成本随攻击者探测次数和时间步增加而减少
                    EU_D_s1 = psi_t_s1 * (-Ra_t) + psi_t_s2 * Rds - CDs
                    EU_D_s2 = psi_t_s2 * (-Ra_t - Rds) + psi_t_s1 * (-CH + Rinf) - CDs
                    #统计数据
                    EU_D_s1_t += EU_D_s1
                    EU_D_s2_t += EU_D_s2
                    EU_D_normal_t += EU_D_s1
                    EU_D_normal_s1_t += EU_D_s1
                    EU_D_normal_s2_t += EU_D_s2
                else:  # 防御者节点类型 θ2 (蜜罐主机)
                    Rdec_t = max(Rdec_initial * np.exp(-lambda_d * num_detected), 0) #防御者欺骗奖励随欺骗成功次数和时间步增加而增加
                    EU_D_s1 = psi_t_s1 * (-CH) + psi_t_s2 * (Rp) - CDs
                    EU_D_s2 = psi_t_s2 * (-CH - Rds) + psi_t_s1 * (-CH + Rdec_t) - CDs
                    #统计数据
                    EU_D_s1_t += EU_D_s1
                    EU_D_s2_t += EU_D_s2
                    EU_D_honeypot_t += EU_D_s2
                    EU_D_honeypot_s1_t += EU_D_s1
                    EU_D_honeypot_s2_t += EU_D_s2
                

                # 防御者根据效用选择信号
                if EU_D_s1 > EU_D_s2:
                    signal = "s1"
                    belief_update = psi_t_s1 #在信号确定之后，信念只有一个
                    normal_signal_nodes.append(i)
                    signal_s1_count += 1
                else:
                    signal = "s2"
                    belief_update = psi_t_s2 #在信号确定之后，信念只有一个
                    signal_s2_count += 1
                    
            # 生成节点属性向量
            X = generate_X(signal)
            X_list.append(X)

            # 累加不同节点类型信念
            if defense_types[i] == 0:
                psi_normal_avg += belief_update
            else:
                psi_honeypot_avg += belief_update

            # 攻击者的行为选择(根据预期效用)
            if signal == "s1":
                EU_A_attack = psi_t_s1 * (Ra_t - Ca1_t - Co) + psi_t_s2 * (-CH + Rinf)
                EU_A_retreat = -Co
                #统计数据
                EU_A_s1_attack_t += EU_A_attack
                EU_A_s1_retreat_t += EU_A_retreat
            else:
                EU_A_attack = psi_t_s2 * (Ra_t - Ca2_t - Co) + psi_t_s1 * (-CH - Rds + Rinf)
                EU_A_retreat = -Co
                #统计数据
                EU_A_s2_attack_t += EU_A_attack
                EU_A_s2_retreat_t += EU_A_retreat

            if EU_A_attack > EU_A_retreat:  # 攻击者选择攻击
                attacker_action = "attack"
                num_probes += 1
                if signal == "s1":
                    #更新攻击者观测事件矩阵
                    action_signal_matrix[0][0] += 1
                    if defense_types[i] == 0:
                        attacker_utility = Ra_t - Ca1_t - Co
                        defender_utility = -Ra_t
                        success_count += 1
                    else:
                        attacker_utility = -Ca3 - Co
                        defender_utility = -CH + Rinf
                        num_detected += 1
                        
                else:
                    #更新攻击者观测事件矩阵
                    action_signal_matrix[0][1] += 1
                    if defense_types[i] == 0:
                        attacker_utility = Ra_t - Ca2_t - Co
                        defender_utility = -Ra_t - CDs
                    else:
                        attacker_utility = -Ca2_t - Co
                        defender_utility = -CH - CDs + Rinf
                        num_detected += 1
            else:  # 攻击者选择撤退
                attacker_action = "retreat"
                if signal == "s1":
                    #更新攻击者观测事件矩阵
                    action_signal_matrix[1][0] += 1
                    if defense_types[i] == 0:
                        attacker_utility = -Co
                        defender_utility = 0
                    else:
                        attacker_utility = -Co + Rp
                        defender_utility = -CH
                else:
                    #更新攻击者观测事件矩阵
                    action_signal_matrix[1][1] += 1
                    if defense_types[i] == 0:
                        attacker_utility = -Co
                        defender_utility = Rds - CDs
                    else:
                        attacker_utility = -Co + Rp
                        defender_utility = -CH - CDs

            # 添加随机噪声
            attacker_utility += np.random.normal(0, noise_scale_attacker)
            defender_utility += np.random.normal(0, noise_scale_defender)

            # # 打印当前节点信息
            # print(f"节点 {i + 1}: 类型 = {'真实系统' if defense_types[i] == 0 else '蜜罐'}, 信号 = {signal}, 攻击者信念 = {beliefs[i]:.3f}, 攻击者行为 = {attacker_action}, 攻击者效用 = {attacker_utility:.3f}, 防御者效用 = {defender_utility:.3f}")

            # 更新信念
            if t > 1:
                new_beliefs[i] = belief_update #在信号确定之后，信念只有一个
            #累加不同信号下的平均信念
            psi_t_s1_avg += psi_t_s1
            psi_t_s2_avg += psi_t_s2
            # 累加效用
            attacker_utility_t += attacker_utility
            defender_utility_t += defender_utility
        #--------------------------------------------节点迭代 end--------------------------------------------

        #-------------------------------------------------------4.更新时间步数据-----------------------------------------------------------------
        # 更新信念
        beliefs = new_beliefs
        #-------------------------------------------------------5.计算和保存统计数据-------------------------------------------------------
        # 5.1保存时间步效用
        attacker_utilities_t.append(attacker_utility_t)
        defender_utilities_t.append(defender_utility_t)

        # 5.2计算EFD和HTD
        efd = calculate_EFD(X_list,normal_signal_nodes)
        htd = calculate_HTD(X_list)
        EFD_t.append(efd)
        HTD_t.append(htd)
        # 5.3保存时间步的统计数据
        success_rates.append(success_count / N)
        # 5.4保存不同信号下的平均信念
        belief_s1_history.append(psi_t_s1_avg/N)
        belief_s2_history.append(psi_t_s2_avg/N)
        # 5.5保存不同节点类型平均信念
        belief_normal_history_avg.append(psi_normal_avg/N)
        belief_honeypot_history_avg.append(psi_honeypot_avg/N)
        # 5.6保存信号计数
        signal_s1_count_history.append(signal_s1_count)
        signal_s2_count_history.append(signal_s2_count)
        # 5.7保存期望效用
        attacker_utility_s1_attack_history.append(EU_A_s1_attack_t/N)
        attacker_utility_s1_retreat_history.append(EU_A_s1_retreat_t/N)
        attacker_utility_s2_attack_history.append(EU_A_s2_attack_t/N)
        attacker_utility_s2_retreat_history.append(EU_A_s2_retreat_t/N)
        defender_utility_s1_history.append(EU_D_s1_t/N)
        defender_utility_s2_history.append(EU_D_s2_t/N)
        defender_utility_normal_history.append(EU_D_normal_t/N)
        defender_utility_honeypot_history.append(EU_D_honeypot_t/N)
        defender_utility_normal_s1_history.append(EU_D_normal_s1_t/N)
        defender_utility_honeypot_s1_history.append(EU_D_honeypot_s1_t/N)
        defender_utility_normal_s2_history.append(EU_D_normal_s2_t/N)
        defender_utility_honeypot_s2_history.append(EU_D_honeypot_s2_t/N)

        
    result = {
        "attacker_utilities_t":attacker_utilities_t, #攻击者效用
        "defender_utilities_t":defender_utilities_t, #防御者效用
        "success_rates":success_rates, #攻击者成功率
        "EFD_t":EFD_t, #曝露虚假程度
        "HTD_t":HTD_t, #隐藏真实程度
        "belief_s1_history":belief_s1_history, #攻击者信念(信号s1)
        "belief_s2_history":belief_s2_history, #攻击者信念(信号s2)
        "belief_normal_history_avg":belief_normal_history_avg, #节点类型正常信念
        "belief_honeypot_history_avg":belief_honeypot_history_avg, #节点类型蜜罐信念
        "signal_s1_count_history":signal_s1_count_history, #信号s1计数
        "signal_s2_count_history":signal_s2_count_history, #信号s2计数
        "attacker_utility_s1_attack_history":attacker_utility_s1_attack_history, #攻击者期望效用(信号s1,攻击)
        "attacker_utility_s1_retreat_history":attacker_utility_s1_retreat_history, #攻击者期望效用(信号s1,撤退)
        "attacker_utility_s2_attack_history":attacker_utility_s2_attack_history, #攻击者期望效用(信号s2,攻击)
        "attacker_utility_s2_retreat_history":attacker_utility_s2_retreat_history, #攻击者期望效用(信号s2,撤退)
        "defender_utility_s1_history":defender_utility_s1_history, #防御者期望效用(信号s1)
        "defender_utility_s2_history":defender_utility_s2_history, #防御者期望效用(信号s2)
        "defender_utility_normal_history":defender_utility_normal_history, #防御者期望效用(节点类型正常)
        "defender_utility_honeypot_history":defender_utility_honeypot_history, #防御者期望效用(节点类型蜜罐)
        "defender_utility_normal_s1_history":defender_utility_normal_s1_history, #防御者期望效用(节点类型正常,信号s1)
        "defender_utility_honeypot_s1_history":defender_utility_honeypot_s1_history, #防御者期望效用(节点类型蜜罐,信号s1)
        "defender_utility_normal_s2_history":defender_utility_normal_s2_history, #防御者期望效用(节点类型正常,信号s2)
        "defender_utility_honeypot_s2_history":defender_utility_honeypot_s2_history #防御者期望效用(节点类型蜜罐,信号s2)
    }
    return result


#--------------1.入参--------------
T=100
N=20


#--------------2.运行模拟--------------
result = run_game_simulation(T,N)
attacker_utilities_t = result["attacker_utilities_t"]
defender_utilities_t = result["defender_utilities_t"]
success_rates = result["success_rates"]
EFD_t = result["EFD_t"]
HTD_t = result["HTD_t"]
belief_s1_history = result["belief_s1_history"]
belief_s2_history = result["belief_s2_history"]
belief_normal_history_avg = result["belief_normal_history_avg"]
belief_honeypot_history_avg = result["belief_honeypot_history_avg"]
signal_s1_count_history = result["signal_s1_count_history"]
signal_s2_count_history = result["signal_s2_count_history"]
#效用相关
attacker_utility_s1_attack_history = result["attacker_utility_s1_attack_history"]
attacker_utility_s1_retreat_history = result["attacker_utility_s1_retreat_history"]
attacker_utility_s2_attack_history = result["attacker_utility_s2_attack_history"]
attacker_utility_s2_retreat_history = result["attacker_utility_s2_retreat_history"]
defender_utility_s1_history = result["defender_utility_s1_history"]
defender_utility_s2_history = result["defender_utility_s2_history"]
defender_utility_normal_history = result["defender_utility_normal_history"]
defender_utility_honeypot_history = result["defender_utility_honeypot_history"]
defender_utility_normal_s1_history = result["defender_utility_normal_s1_history"]
defender_utility_honeypot_s1_history = result["defender_utility_honeypot_s1_history"]
defender_utility_normal_s2_history = result["defender_utility_normal_s2_history"]
defender_utility_honeypot_s2_history = result["defender_utility_honeypot_s2_history"]
#--------------3.绘制结果--------------
# 横排三幅子图
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: 攻击者和防御者效用随时间变化
axs[0].plot(range(1, T + 1), attacker_utilities_t, label="Attacker Utility", color="blue")
axs[0].plot(range(1, T + 1), defender_utilities_t, label="Defender Utility", color="orange")
axs[0].legend()
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("Utility")
axs[0].set_title("Utility Over Time")

# 子图2: 攻击者成功率随时间变化
axs[1].plot(range(1, T + 1), success_rates, label="Attacker Success Rate", color="purple")
axs[1].legend()
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("Success Rate")
axs[1].set_title("Attacker Success Rate Over Time")

# 子图3: EFD 和 HTD 随时间变化
axs[2].plot(range(1, T + 1), EFD_t, label="EFD (Exposure of Falsehood Degree)", color="blue")
axs[2].plot(range(1, T + 1), HTD_t, label="HTD (Hiding of True Degree)", color="green")
axs[2].legend()
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Metrics")
axs[2].set_title("EFD and HTD Over Time")

plt.tight_layout()
plt.savefig("combined_plots_high_dpi.png", dpi=400)
plt.show()



# 3.4-3.6 绘制攻击者信念、节点类型信念和信号计数随时间变化的曲线
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 子图1: 攻击者信念(信号s1,s2)随时间变化
axs[0].plot(range(1, T + 1), belief_s1_history, label="Attacker Belief (Signal s1)", color="blue")
axs[0].plot(range(1, T + 1), belief_s2_history, label="Attacker Belief (Signal s2)", color="green")
axs[0].legend()
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("Belief")
axs[0].set_title("Attacker Belief Over Time")

# 子图2: 节点类型信念随时间变化
axs[1].plot(range(1, T + 1), belief_normal_history_avg, label="Node Type Normal Belief", color="blue")
axs[1].plot(range(1, T + 1), belief_honeypot_history_avg, label="Node Type Honeypot Belief", color="green")
axs[1].legend()
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("Belief")
axs[1].set_title("Node Type Belief Over Time")

# 子图3: 信号s1,s2计数随时间变化
axs[2].plot(range(1, T + 1), signal_s1_count_history, label="Signal s1 Count", color="blue")
axs[2].plot(range(1, T + 1), signal_s2_count_history, label="Signal s2 Count", color="green")
axs[2].legend()
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Count")
axs[2].set_title("Signal s1 and s2 Count Over Time")

plt.tight_layout()
plt.show()

# 3.7-3.10 绘制期望效用  
fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# 子图1: 攻击者期望效用(信号s1,s2,攻击,撤退)随时间变化
axs[0].plot(range(1, T + 1), attacker_utility_s1_attack_history, label="Attacker Utility (Signal s1, Attack)", color="blue")
axs[0].plot(range(1, T + 1), attacker_utility_s1_retreat_history, label="Attacker Utility (Signal s1, Retreat)", color="green")
axs[0].plot(range(1, T + 1), attacker_utility_s2_attack_history, label="Attacker Utility (Signal s2, Attack)", color="red")
axs[0].plot(range(1, T + 1), attacker_utility_s2_retreat_history, label="Attacker Utility (Signal s2, Retreat)", color="purple")
axs[0].legend()
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("Utility")
axs[0].set_title("Attacker Utility Over Time")

# 子图2: 防御者期望效用(信号s1,s2)随时间变化
axs[1].plot(range(1, T + 1), defender_utility_s1_history, label="Defender Utility (Signal s1)", color="blue")
axs[1].plot(range(1, T + 1), defender_utility_s2_history, label="Defender Utility (Signal s2)", color="green")
axs[1].legend()
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("Utility")
axs[1].set_title("Defender Utility Over Time")

# 子图3: 防御者期望效用(节点类型正常,蜜罐)随时间变化
axs[2].plot(range(1, T + 1), defender_utility_normal_history, label="Defender Utility (Node Type Normal)", color="blue")
axs[2].plot(range(1, T + 1), defender_utility_honeypot_history, label="Defender Utility (Node Type Honeypot)", color="green")
axs[2].legend()
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("Utility")
axs[2].set_title("Defender Utility Over Time")

# 子图4: 防御者期望效用(节点类型正常,蜜罐,信号s1,s2)随时间变化
axs[3].plot(range(1, T + 1), defender_utility_normal_s1_history, label="Defender Utility (Node Type Normal, Signal s1)", color="blue")
axs[3].plot(range(1, T + 1), defender_utility_honeypot_s1_history, label="Defender Utility (Node Type Honeypot, Signal s1)", color="green")
axs[3].plot(range(1, T + 1), defender_utility_normal_s2_history, label="Defender Utility (Node Type Normal, Signal s2)", color="red")
axs[3].plot(range(1, T + 1), defender_utility_honeypot_s2_history, label="Defender Utility (Node Type Honeypot, Signal s2)", color="purple")
axs[3].legend()
axs[3].set_xlabel("Time Step")
axs[3].set_ylabel("Utility")
axs[3].set_title("Defender Utility Over Time")

plt.tight_layout()
plt.show()
