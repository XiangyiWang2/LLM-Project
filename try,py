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
        self.feature_dim = feature_dim
        self.num_arms = num_arms
        self.noise = noise
        self.theta_star = torch.randn(feature_dim, device=device)
        self.theta_star /= torch.norm(self.theta_star)

    def generate_context(self):
        """为当前回合生成一组臂的特征向量"""
        # 随机生成归一化的特征向量
        context = torch.randn(self.num_arms, self.feature_dim, device=device)
        context /= torch.norm(context, dim=1, keepdim=True)
        return context

    def get_preference(self, arm1_features, arm2_features):
        """根据真实模型 theta_star 和噪声，模拟用户偏好"""
        true_utility1 = torch.dot(arm1_features, self.theta_star)
        true_utility2 = torch.dot(arm2_features, self.theta_star)
        
        # 使用 Logistic 模型来决定获胜概率
        prob_arm1_wins = 1.0 / (1.0 + torch.exp(-self.noise * (true_utility1 - true_utility2)))
        
        # 根据概率进行二项分布采样，返回1代表arm1赢，0代表arm2赢
        return 1 if np.random.binomial(1, prob_arm1_wins.cpu().item()) else 0

    def calculate_regret(self, context, chosen_arm1_features, chosen_arm2_features):
        """计算当前回合的后悔度"""
        true_utilities = context @ self.theta_star
        best_arm_idx = torch.argmax(true_utilities)
        best_arm_features = context[best_arm_idx]
        
        # 理想情况是选择最优的臂和另一个臂进行比较
        optimal_utility = torch.dot(best_arm_features, self.theta_star)
        chosen_utility = torch.dot(chosen_arm1_features, self.theta_star)
        
        # Regret 定义为最优臂的效用与所选臂（通常是arm1）的效用之差
        return (optimal_utility - chosen_utility).item()


# ==============================================================================
# 部分 2: 算法 (The LDB-Delay Agent)
# ==============================================================================
class LDB_Delay_Agent:
    def __init__(self, feature_dim, lambda_reg):
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg

        # 算法初始化 (对应 Line 2, 3, 4)
        self.theta = torch.zeros(feature_dim, device=device)
        self.V = lambda_reg * torch.eye(feature_dim, device=device)
        self.arrived_history = []  # H_t: 存储 (z, y)

    def select_arms(self, context, t, delta):
        """根据当前模型和置信区间选择两个臂 (对应 Line 9, 10)"""
        # 计算 beta_t (置信区间的宽度)
        # 注意：这里的 L 假设为1 (特征向量已归一化), kappa_mu 也是常数
        # Convert all Python numbers to tensors before PyTorch operations
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

        if arm2_idx == -1: # 如果所有臂都一样，随便选一个不同的
            arm2_idx = 1 if arm1_idx == 0 else 0
        arm2_features = context[arm2_idx]
        
        return arm1_idx, arm2_idx

    def update_V_matrix(self, arm1_features, arm2_features):
        """立即更新协方差矩阵 V (对应 Line 12)"""
        z_t = arm1_features - arm2_features
        self.V += torch.outer(z_t, z_t)
        return z_t

    def receive_feedback(self, z_t, winner):
        """接收到达的反馈并更新历史记录 (对应 Line 6, 7)"""
        self.arrived_history.append({'z': z_t, 'y': torch.tensor(winner, device=device, dtype=torch.float32)})
    
    def update_theta(self):
        """基于所有已到达的历史数据，更新模型参数 theta (对应 Line 8)"""
        if not self.arrived_history:
            return # 如果没有任何历史数据，则不更新

        # 我们使用 LBFGS 优化器来最小化负对数似然损失
        theta_to_optimize = self.theta.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([theta_to_optimize], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = 0.5 * self.lambda_reg * torch.norm(theta_to_optimize)**2
            for item in self.arrived_history:
                z = item['z']
                y = item['y']
                score = torch.dot(z, theta_to_optimize)
                # Binary Cross-Entropy with Logits Loss
                loss += - (y * score - torch.log(1 + torch.exp(score)))
            loss.backward()
            return loss

        optimizer.step(closure)
        self.theta = theta_to_optimize.detach()


# ==============================================================================
# 部分 3: 实验主循环 (Main Experiment Loop)
# ==============================================================================
if __name__ == '__main__':
    # --- 实验配置 ---
    NUM_ITERATIONS = 10000
    FEATURE_DIM = 10
    NUM_ARMS = 20
    LAMBDA_REG = 1.0
    NOISE = 1.0
    DELTA = 0.1 # 置信度参数
    
    # LDB-Delay 核心参数
    DELAY_WINDOW_M = 50   # 最大延迟窗口
    OBS_PROB_RHO = 0.9    # 反馈被观察到的概率

    # --- 初始化 ---
    env = LinearDuelingBanditEnvironment(FEATURE_DIM, NUM_ARMS, NOISE)
    agent = LDB_Delay_Agent(FEATURE_DIM, LAMBDA_REG)
    
    pending_feedback_buffer = [] # (arrival_time, z_t, winner)
    cumulative_regret = 0.0
    regret_history = []

    # --- 主循环 ---
    for t in tqdm(range(1, NUM_ITERATIONS + 1)):
        # 1. 检查并分发已到达的反馈 (Line 6, 7)
        remaining_buffer = []
        for arrival_time, z_t, winner in pending_feedback_buffer:
            if t >= arrival_time:
                agent.receive_feedback(z_t, winner)
            else:
                remaining_buffer.append((arrival_time, z_t, winner))
        pending_feedback_buffer = remaining_buffer

        # 2. 更新模型参数 theta (Line 8)
        agent.update_theta()

        # 3. 生成上下文并选择臂
        context = env.generate_context()
        arm1_idx, arm2_idx = agent.select_arms(context, t, DELTA)
        arm1_features, arm2_features = context[arm1_idx], context[arm2_idx]

        # 4. 立即更新 V 矩阵 (Line 12)
        z_t = agent.update_V_matrix(arm1_features, arm2_features)

        # 5. 从环境获取真实反馈 (但Agent还不知道)
        winner = env.get_preference(arm1_features, arm2_features)
        
        # 6. 模拟延迟，将反馈放入缓冲池 (Line 11)
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
    plt.xlabel("Iteration (t)")
    plt.ylabel("Cumulative Regret")
    plt.title(f"LDB-Delay Performance (m={DELAY_WINDOW_M}, ρ={OBS_PROB_RHO})")
    plt.grid(True)
    plt.show()
