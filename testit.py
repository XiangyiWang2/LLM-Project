import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 使用 GPU (如果可用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 部分 1: 环境 (Environment)
# ==============================================================================
class LinearDuelingBanditEnvironment:
    def __init__(self, feature_dim, num_arms, noise=1.0):
        self.feature_dim = feature_dim # 特征维度
        self.num_arms = num_arms # 臂的数量
        self.noise = noise # 噪声
        self.theta_star = torch.randn(feature_dim, device=device) # 生成真实的 theta*
        self.theta_star /= torch.norm(self.theta_star) # 归一化

    def generate_context(self):
        """为单轮生成一组臂（上下文）"""
        context = torch.randn(self.num_arms, self.feature_dim, device=device)
        context /= torch.norm(context, dim=1, keepdim=True) # 对每个臂的特征向量进行归一化
        return context

    def get_preference(self, arm1_features, arm2_features):
        """基于 Logistic 模型决定决斗的胜者"""
        true_utility1 = torch.dot(arm1_features, self.theta_star)
        true_utility2 = torch.dot(arm2_features, self.theta_star)
        
        prob_arm1_wins = 1.0 / (1.0 + torch.exp(-self.noise * (true_utility1 - true_utility2)))
        
        # 如果臂1赢了返回1，否则返回0
        return 1 if np.random.binomial(1, prob_arm1_wins.cpu().item()) else 0

    def calculate_regret(self, context, chosen_arm1_features, chosen_arm2_features):
        """计算当前回合的后悔度"""
        true_utilities = context @ self.theta_star
        best_arm_idx = torch.argmax(true_utilities) # 找到最优臂的索引
        best_arm_features = context[best_arm_idx] # 最优臂的特征
        
        optimal_utility = torch.dot(best_arm_features, self.theta_star) # 最优臂的效用
        # 后悔度通常是相对于最优臂定义的，在策略的'贪心'部分，最优臂是 arm1
        chosen_utility = torch.dot(chosen_arm1_features, self.theta_star) # 被选中臂的效用
        
        return (optimal_utility - chosen_utility).item()


# ==============================================================================
# 部分 2: LDB-Delay 智能体
# ==============================================================================
class LDB_Delay_Agent:
    def __init__(self, feature_dim, lambda_reg, obs_prob_rho):
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        # 修改: 存储观测概率 rho
        self.rho = obs_prob_rho

        # 算法初始化 (对应伪代码 Line 2, 3, 4)
        self.theta = torch.zeros(feature_dim, device=device)
        self.V = lambda_reg * torch.eye(feature_dim, device=device)
        self.arrived_history = []  # H_t: 存储已到达的反馈 (z, y)

    def select_arms(self, context, t, delta):
        """根据当前模型和置信区间选择两个臂 (对应伪代码 Line 9, 10)"""
        # 计算 beta_t (置信区间的宽度)
        # 注意: 为简单起见, 假设 L=1 (特征已归一化), kappa_mu=1
        term1 = torch.sqrt(torch.tensor(self.lambda_reg, device=device))
        log_term1 = torch.log(torch.tensor(1 / delta, device=device))
        log_term2_inner = 1 + t / (self.feature_dim * self.lambda_reg)
        log_term2 = torch.log(torch.tensor(log_term2_inner, device=device))
        term2 = torch.sqrt(2 * log_term1 + self.feature_dim * log_term2)
        beta_t = term1 + term2

        # 1. 选择第一个臂 (Line 9)
        estimated_utilities = context @ self.theta
        arm1_idx = torch.argmax(estimated_utilities).item()
        arm1_features = context[arm1_idx]

        # 2. 选择第二个臂 (Line 10)
        max_ucb = -float('inf')
        arm2_idx = -1
        
        V_inv = torch.linalg.inv(self.V)
        for i, arm_features in enumerate(context):
            if i == arm1_idx:
                continue
            
            diff = arm_features - arm1_features
            ucb_score = torch.dot(self.theta, diff) + beta_t * torch.sqrt(torch.dot(diff, V_inv @ diff))
            
            if ucb_score > max_ucb:
                max_ucb = ucb_score
                arm2_idx = i

        if arm2_idx == -1: # 如果所有臂都相同，则备用选择
            arm2_idx = 1 if arm1_idx == 0 else 0
        arm2_features = context[arm2_idx]
        
        return arm1_idx, arm2_idx

    def update_V_matrix(self, arm1_features, arm2_features):
        """立即更新协方差矩阵 V (对应伪代码 Line 12)"""
        z_t = arm1_features - arm2_features
        self.V += torch.outer(z_t, z_t)
        return z_t

    def receive_feedback(self, z_t, winner):
        """接收到达的反馈并更新历史记录 (对应伪代码 Line 6, 7)"""
        self.arrived_history.append({'z': z_t, 'y': torch.tensor(winner, device=device, dtype=torch.float32)})
    
    def update_theta(self):
        """基于所有已到达的历史数据，更新模型参数 theta (对应伪代码 Line 8)"""
        if not self.arrived_history:
            return

        # 我们使用 LBFGS 优化器来最小化论文中的负对数似然损失
        theta_to_optimize = self.theta.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([theta_to_optimize], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            # 从正则化项开始: 0.5 * lambda * ||theta'||^2
            loss = 0.5 * self.lambda_reg * torch.norm(theta_to_optimize)**2
            
            # 对历史记录 H_t 中所有收到的反馈的损失进行求和
            for item in self.arrived_history:
                z = item['z']
                y = item['y']
                score = torch.dot(z, theta_to_optimize)
                
                # ======================================================================
                # 修改: 这里现在实现了 L_t(theta') 损失函数
                # 论文中对于单个观测 (s, y_s) 的损失项是:
                # -[ (y_s/rho) * log(mu(score)) + (1 - y_s/rho) * log(1-mu(score)) ]
                # 为了数值稳定性，可以重写为:
                # log(1 + exp(score)) - (y_s/rho) * score
                # 这等价于 PyTorch 的 BCEWithLogitsLoss，但使用了加权标签 y_s/rho。
                # ======================================================================
                loss += torch.log(1 + torch.exp(score)) - (y / self.rho) * score

            loss.backward()
            return loss

        optimizer.step(closure)
        self.theta = theta_to_optimize.detach()


# ==============================================================================
# 部分 3: 实验主循环
# ==============================================================================
if __name__ == '__main__':
    # --- 实验配置 ---
    NUM_ITERATIONS = 2000
    FEATURE_DIM = 10
    NUM_ARMS = 20
    LAMBDA_REG = 1.0
    NOISE = 1.0
    DELTA = 0.1
    
    # LDB-Delay 核心参数
    DELAY_WINDOW_M = 50
    OBS_PROB_RHO = 0.9

    # --- 初始化 ---
    env = LinearDuelingBanditEnvironment(FEATURE_DIM, NUM_ARMS, NOISE)
    # 修改: 将 OBS_PROB_RHO 传递给智能体
    agent = LDB_Delay_Agent(FEATURE_DIM, LAMBDA_REG, OBS_PROB_RHO)
    
    pending_feedback_buffer = [] # 存储 (arrival_time, z_t, winner)
    cumulative_regret = 0.0
    regret_history = []

    # --- 主循环 ---
    for t in tqdm(range(1, NUM_ITERATIONS + 1), desc="运行 LDB-Delay"):
        # 1. 检查并分发新到达的反馈 (Lines 6, 7)
        remaining_buffer = []
        for arrival_time, z_t, winner in pending_feedback_buffer:
            if t >= arrival_time:
                agent.receive_feedback(z_t, winner)
            else:
                remaining_buffer.append((arrival_time, z_t, winner))
        pending_feedback_buffer = remaining_buffer

        # 2. 基于所有历史 H_t 更新模型参数 theta (Line 8)
        if agent.arrived_history: # 仅在有反馈时更新
             agent.update_theta()

        # 3. 生成上下文并选择臂 (Lines 9, 10)
        context = env.generate_context()
        arm1_idx, arm2_idx = agent.select_arms(context, t, DELTA)
        arm1_features, arm2_features = context[arm1_idx], context[arm2_idx]

        # 4. 立即更新 V 矩阵 (Line 12)
        z_t = agent.update_V_matrix(arm1_features, arm2_features)

        # 5. 从环境中获取真实反馈 (智能体此时还不知道)
        winner = env.get_preference(arm1_features, arm2_features)
        
        # 6. 模拟延迟和观测概率 (Line 11)
        if np.random.rand() < OBS_PROB_RHO:
            delay = np.random.randint(1, DELAY_WINDOW_M + 1)
            arrival_time = t + delay
            pending_feedback_buffer.append((arrival_time, z_t, winner))
        
        # 7. 记录后悔度
        regret = env.calculate_regret(context, arm1_features, arm2_features)
        cumulative_regret += regret
        regret_history.append(cumulative_regret)

    # --- 结果可视化 ---
    plt.figure(figsize=(10, 6))
    plt.plot(regret_history)
    plt.xlabel("迭代次数 (t)")
    plt.ylabel("累积后悔度")
    plt.title(f"LDB-Delay 性能 (m={DELAY_WINDOW_M}, ρ={OBS_PROB_RHO})")
    plt.grid(True)
    plt.show() 