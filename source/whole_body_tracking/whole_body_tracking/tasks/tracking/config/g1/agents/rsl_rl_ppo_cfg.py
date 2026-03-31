from isaaclab.utils import configclass

# IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/
# algorithm: RslRlPpoAlgorithmCfg, RslRlDistillationAlgorithmCfg
# policy: RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, 
    RslRlPpoActorCriticCfg, 
    RslRlPpoAlgorithmCfg, 
    RslRlDistillationAlgorithmCfg
)

# PAMR: Stage 1 Training
from whole_body_tracking.utils.supervise import (
    SuperviseTrainer
)

# policy: RslRlDistillationCfg
from whole_body_tracking.utils.rsl_rl_cfg import (
    RslRlPpoActorCriticTransformerCfg, 
    RslRlPpoActorCriticFSQCfg, 
    RslRlPpoActorCriticVQCfg,
    RslRlPpoActorCriticAttentionCfg,
    RslRlDistillationCfg,
    RslRlSuperviseJointPosCfg, # PAMR: Stage 1 Training
    RslRlFrontEndResidualActorCriticCfg, # PAMR: Stage 2 RL Finetuning
)

@configclass
class RslRlSuperviseAlgorithmCfg:
    """Configuration for the supervised learning algorithm."""

    class_name: str = "SuperviseTrainer"
    """The algorithm class name. Defaults to SuperviseTrainer."""
    num_learning_epochs: int = 5
    learning_rate: float = 1.0e-3
    gradient_length: int = 15
    max_grad_norm: float = 1.0
    loss_type: str = "mse"

@configclass
class G1FlatSupervisedRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 2
    experiment_name = "g1_flat_supervised"
    empirical_normalization = True

    from pathlib import Path

    path1 = Path("/home/yuxuancheng/MOSAIC/model/exported/policy.onnx")
    path2 = Path("/home/chengyuxuan/MOSAIC/model/exported/policy.onnx")

    model_path = path1 if path1.exists() else (path2 if path2.exists() else None)
    
    policy = RslRlSuperviseJointPosCfg( # from rsl_rl_cfg.py
        class_name="SuperviseLearning",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        gmt_path=model_path,) # <--- 在这里传入预训练的 GMT ONNX 模型路径

    algorithm = RslRlSuperviseAlgorithmCfg()

@configclass
class G1FlatFrontRESFinetuneRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for Stage 2: RL Finetuning of FrontRES."""
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 500
    experiment_name = "g1_flat_frontres_finetune"
    empirical_normalization = True
    resume = True # <-- 必须设置为 True 来加载阶段一的模型

    policy = RslRlFrontEndResidualActorCriticCfg(
        class_name="FrontEndResidualActorCritic",
        # FrontRES 结构应与阶段一监督学习的 student_hidden_dims 保持一致
        residual_hidden_dims=[1024, 1024, 512, 256], 
        # !! 关键 !!: 需要根据你的观测定义, 准确填写 q_ref 在 obs 向量中的起始索引
        # 例如, 如果 obs = [base_vel(3), base_ang_vel(3), q_ref(29), ...], 则 q_ref_start_idx = 3 + 3 = 6
        q_ref_start_idx= 0, # 假设 q_ref (command) 在最前面
        init_noise_std=0.1, # 微调阶段给定一个较小的初始探索噪声
        gmt_checkpoint_path="/path/to/your/gmt_model.pt", # 这里需要放 GMT 的 Pytorch Checkpoint
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005, # 鼓励探索
        learning_rate=5.0e-4, # 较小的微调学习率
    )

@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 1000
    experiment_name = "g1_flat_mosaic_hybrid"
    # experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",)

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,)


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)

@configclass
class G1FlatDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True
    policy = RslRlDistillationCfg(
        class_name="StudentTeacher",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        teacher_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",)

    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="Distillation",
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length = 15)
    

@configclass
class G1FlatKLDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Configuration for KL-based distillation (improved version).

    This uses KL divergence loss instead of MSE, matching MOSAIC's approach.
    Expected to provide better imitation performance than standard distillation.
    """
    num_steps_per_env = 24
    max_iterations = 200000
    save_interval = 500
    experiment_name = "g1_flat_kl_distillation"
    empirical_normalization = True

    policy = RslRlDistillationCfg(
        class_name="StudentTeacher",
        init_noise_std=1.0,
        student_hidden_dims=[1024, 1024, 512, 256],
        teacher_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",)
