import torch
from torch import nn

from mixture_of_experts import MoE, HeirarchicalMoE



# 定义一个专家混合模型（Mixture of Experts, MoE）
# 策略选项：
    # 'all' - 总是使用第二个专家
    # 'none' - 从不使用第二个专家
    # 'threshold' - 如果门控值大于给定阈值，则使用第二个专家
    # 'random' - 如果门控值大于阈值乘以一个随机数（范围在0到1之间），则使用第二个专家
moe = MoE(
    dim=512,  # 输入数据的特征维度
    num_experts=16,  # 专家的数量。增加专家数量可以在不增加计算量的情况下增加模型参数
    hidden_dim=512 * 4,  # 每个专家中隐藏层的维度，默认为输入维度的4倍
    activation=nn.LeakyReLU,  # 使用你偏好的激活函数，这里使用 LeakyReLU。如果未指定，将默认使用 GELU
    second_policy_train='random',  # 在 Top-2 门控中，选择第二个专家的策略。训练阶段使用 'random' 策略
    second_policy_eval='random',  # 在 Top-2 门控中，选择第二个专家的策略。评估阶段使用 'random' 策略
    second_threshold_train=0.2,  # 训练阶段选择第二个专家的阈值
    second_threshold_eval=0.2,  # 评估阶段选择第二个专家的阈值
    capacity_factor_train=1.25,  # 专家在每个批次中有固定的容量。为了防止门控不平衡，需要一些额外的容量
    capacity_factor_eval=2.,  # capacity_factor_* 应该设置为一个大于或等于1的值
    loss_coef=1e-2  # 辅助专家平衡辅助损失的乘数
)

# 创建一个随机输入张量，形状为 (4, 1024, 512)
inputs = torch.randn(4, 1024, 512)

# 前向传播，通过 MoE 模型处理输入数据
# 输出 out 的形状为 (4, 1024, 512)
# 辅助损失 aux_loss 的形状为 (1,)
out, aux_loss = moe(inputs)

# 输出结果
print(f"输出张量形状: {out.shape}")
print(f"辅助损失: {aux_loss.item()}")



# 定义一个双层专家混合模型（Hierarchical Mixture of Experts, HMoE）
# 参数说明:
#   dim = 512: 输入数据的特征维度
#   num_experts = (4, 4): 第一层有4个门控，每个门控下有4个专家，总共16个专家
moe = HeirarchicalMoE(
    dim = 512,
    num_experts = (4, 4), 
)

# 创建一个随机输入张量，形状为 (4, 1024, 512)
inputs = torch.randn(4, 1024, 512)

# 前向传播，通过 HMoE 模型处理输入数据
# 输出 out 的形状为 (4, 1024, 512)
# 辅助损失 aux_loss 的形状为 (1,)
out, aux_loss = moe(inputs)

# 输出结果
print(f"输出张量形状: {out.shape}")
print(f"辅助损失: {aux_loss.item()}")



# 另一种 HMoE 实例化方式
# 参数说明:
#   dim = 512: 输入数据的特征维度
#   num_experts = (22, 22): 第一层有22个门控，每个门控下有22个专家，总共484个专家
moe = HeirarchicalMoE(
    dim = 512,
    num_experts = (22, 22)
).cuda()

# 创建一个随机输入张量，并将其移动到GPU上
inputs = torch.randn(1, 1024, 512).cuda()

# 前向传播，通过 HMoE 模型处理输入数据
out, aux_loss = moe(inputs)

# 计算模型的总参数量
total_params = sum(p.numel() for p in moe.parameters())
print(f'number of parameters - {total_params}')



# 定义一个三层多层感知机（MLP）作为专家网络
class Experts(nn.Module):
    """
    Experts 类实现了一个三层多层感知机（MLP）作为专家网络。
    每个专家由三个线性层组成，中间使用 LeakyReLU 激活函数。

    参数说明:
        dim (int): 输入和输出的特征维度。
        num_experts (int, 可选): 专家的数量，默认为16。
    """
    def __init__(self, dim, num_experts = 16):
        super().__init__()
        # 初始化第一层线性变换的权重，形状为 (num_experts, dim, dim * 4)
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 4))
        # 初始化第二层线性变换的权重，形状为 (num_experts, dim * 4, dim * 4)
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 4, dim * 4))
        # 初始化第三层线性变换的权重，形状为 (num_experts, dim * 4, dim)
        self.w3 = nn.Parameter(torch.randn(num_experts, dim * 4, dim))
        # 定义激活函数，这里使用 LeakyReLU，并启用 inplace 操作以节省内存
        self.act = nn.LeakyReLU(inplace = True)

    def forward(self, x):
        """
        前向传播方法，对输入数据应用三层 MLP。

        参数:
            x (torch.Tensor): 输入张量，形状为 (num_experts, batch_size, dim)。

        返回:
            torch.Tensor: 输出张量，形状为 (num_experts, batch_size, dim)。
        """
        # 第一层线性变换后应用激活函数
        hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        # 第二层线性变换后应用激活函数
        hidden2 = self.act(torch.einsum('end,edh->enh', hidden1, self.w2))
        # 第三层线性变换得到最终输出
        out = torch.einsum('end,edh->enh', hidden2, self.w3)
        return out

# 实例化专家网络，假设输入维度为512，专家数量为16
experts = Experts(512, num_experts = 16)

# 实例化 MoE 模型，使用上述定义的专家网络
moe = MoE(
    dim=512,  # 输入数据的特征维度
    num_experts=16,  # 专家的数量
    experts=experts  # 指定的专家网络
)

# 创建一个随机输入张量，形状为 (4, 1024, 512)
inputs = torch.randn(4, 1024, 512)

# 前向传播，通过 MoE 模型处理输入数据
# 输出 out 的形状为 (4, 1024, 512)
# 辅助损失 aux_loss 的形状为 (1,)
out, aux_loss = moe(inputs)

# 输出结果
print(f"输出张量形状: {out.shape}")
print(f"辅助损失: {aux_loss.item()}")
