import torch

class ParabolicAccelerationFilter:
    def __init__(self, window_size=21, dt=0.02, device='cpu'):
        """
        初始化时，一次性把数学里最繁琐的矩阵求逆做完，存为常数。
        """
        self.window_size = window_size
        
        # 1. 生成相对时间序列 t [-0.2, -0.18, ..., 0, ..., 0.18, 0.2]
        half_window = window_size // 2
        t = torch.arange(-half_window, half_window + 1, dtype=torch.float32, device=device) * dt
        
        # 2. 构建设计矩阵 X (大小为 window_size x 3)
        # 第一列是 t^2, 第二列是 t, 第三列是 1
        X = torch.stack([t**2, t, torch.ones_like(t)], dim=1)
        
        # 3. 计算伪逆矩阵 M = (X^T * X)^-1 * X^T
        # torch.linalg.pinv 直接计算伪逆矩阵
        M = torch.linalg.pinv(X) 
        
        # 4. 提取矩阵的第一行 (这就是专门用来算 A 的常数权重向量!)
        self.weight_A = M[0, :] # shape: (window_size,)
        
    def get_az(self, z_sequence):
        """
        在线运行时调用，速度快到飞起（只做一次一维点乘）
        z_sequence shape: (batch_size, window_size)
        """
        # 抛物线系数 A = 权重向量 点乘 Z高度序列
        A = torch.matmul(z_sequence, self.weight_A)
        
        # 物理加速度 a_z = 2 * A
        a_z = 2.0 * A
        
        return a_z

if __name__ == "__main__":
    # ====== 使用示例 ======
    filter = ParabolicAccelerationFilter(window_size=21, dt=0.02)
    # 假设你在 RL 的 step 里拿到了过去和未来 21 帧的 Z 轴数据
    z_seq = torch.rand(1, 21) # 假数据
    az = filter.get_az(z_seq)
    print(f"提取出的加速度 a_z: {az.item():.2f}")