import torch
import numpy as np
import matplotlib
# 方案一: 保存图片时不需要TkAgg
# matplotlib.use('TkAgg') 
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
        context = torch.randn(self.num_arms, self.feature_dim, device=device)
        context /= torch.norm(context, dim=1, keepdim=True)
        return context

    def get_preference(self, arm1_features, arm2_features):
        true_utility1 = torch.dot(arm1_features, self.theta_star)
        true_utility2 = torch.dot(arm2_features, self.theta_star)
        prob_arm1_wins = 1.0 / (1.0 + torch.exp(-self.noise * (true_utility1 - true_utility2)))
        
        # 微调: 减少 CPU-GPU 同步
        return torch.bernoulli(prob_arm1_wins).item()

    def calculate_regret(self, context, chosen_arm1_features, chosen_arm2_features):
        with torch.no_grad():
            true_utilities = context @ self.theta_star
            best_arm_idx = torch.argmax(true_utilities)
            best_arm_features = context[best_arm_idx]
            optimal_utility = torch.dot(best_arm_features, self.theta_star)
            chosen_utility = torch.dot(chosen_arm1_features, self.theta_star)
            return (optimal_utility - chosen_utility).item()

# ==============================================================================
# 部分 2: LDB-Delay 智能体 (部分优化, 每轮更新)
# ==============================================================================
class LDB_Delay_Agent_Heavy_Update:
    def __init__(self, feature_dim, lambda_reg, obs_prob_rho):
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.rho = obs_prob_rho

        self.theta = torch.zeros(feature_dim, device=device)
        self.V = lambda_reg * torch.eye(feature_dim, device=device)
        self.arrived_history = []
        
        # 微调: 维护 V 的 Cholesky 分解 L，避免求逆
        self.L = torch.linalg.cholesky(self.V)

    def select_arms(self, context, t, delta):
        """微调: 此函数已被向量化，以避免循环和矩阵求逆"""
        with torch.no_grad():
            term1 = torch.sqrt(torch.tensor(self.lambda_reg, device=device))
            log_term1 = torch.log(torch.tensor(1 / delta, device=device))
            log_term2_inner = 1 + t / (self.feature_dim * self.lambda_reg)
            log_term2 = torch.log(torch.tensor(log_term2_inner, device=device))
            term2 = torch.sqrt(2 * log_term1 + self.feature_dim * log_term2)
            beta_t = term1 + term2

            estimated_utilities = context @ self.theta
            arm1_idx = torch.argmax(estimated_utilities)
            arm1_features = context[arm1_idx]
            
            diffs = context - arm1_features
            sol = torch.cholesky_solve(diffs.T, self.L).T
            quad_forms = torch.sum(diffs * sol, dim=1)
            ucb_scores = (diffs @ self.theta) + beta_t * torch.sqrt(quad_forms)
            ucb_scores[arm1_idx] = -float('inf')
            arm2_idx = torch.argmax(ucb_scores)
            
        return arm1_idx.item(), arm2_idx.item()

    def update_V_matrix(self, arm1_features, arm2_features):
        z_t = arm1_features - arm2_features
        self.V += torch.outer(z_t, z_t)
        # 微调: V 更新后，重新计算 Cholesky 分解
        self.L = torch.linalg.cholesky(self.V)
        return z_t

    def receive_feedback(self, z_t, winner):
        self.arrived_history.append({'z': z_t, 'y': torch.tensor(winner, device=device, dtype=torch.float32)})
    
    def update_theta(self):
        """!!!遵从您的要求，此函数内部逻辑未被修改!!!"""
        if not self.arrived_history:
            return

        theta_to_optimize = self.theta.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([theta_to_optimize], max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = 0.5 * self.lambda_reg * torch.norm(theta_to_optimize)**2
            for item in self.arrived_history:
                z = item['z']
                y = item['y']
                score = torch.dot(z, theta_to_optimize)
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
    DELAY_WINDOW_M = 50
    OBS_PROB_RHO = 0.9

    # --- 初始化 ---
    env = LinearDuelingBanditEnvironment(FEATURE_DIM, NUM_ARMS, NOISE)
    agent = LDB_Delay_Agent_Heavy_Update(FEATURE_DIM, LAMBDA_REG, OBS_PROB_RHO)
    
    pending_feedback_buffer = []
    cumulative_regret = 0.0
    regret_history = []

    # --- 主循环 ---
    for t in tqdm(range(1, NUM_ITERATIONS + 1), desc="运行LDB-Delay(每轮更新)"):
        # 1. 检查并分发新到达的反馈
        remaining_buffer = []
        for arrival_time, z_t, winner in pending_feedback_buffer:
            if t >= arrival_time:
                agent.receive_feedback(z_t, winner)
            else:
                remaining_buffer.append((arrival_time, z_t, winner))
        pending_feedback_buffer = remaining_buffer

        # 2. 修改回每轮都更新 theta
        if agent.arrived_history:
             agent.update_theta()

        # 3. 生成上下文并选择臂
        context = env.generate_context()
        arm1_idx, arm2_idx = agent.select_arms(context, t, DELTA)
        arm1_features, arm2_features = context[arm1_idx], context[arm2_idx]

        # 4. 立即更新 V 矩阵
        z_t = agent.update_V_matrix(arm1_features, arm2_features)

        # 5. 从环境中获取真实反馈
        winner = env.get_preference(arm1_features, arm2_features)
        
        # 6. 模拟延迟和观测概率
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
    plt.savefig("regret_plot_heavy_update.png")
    print("\n绘图已保存为 regret_plot_heavy_update.png")