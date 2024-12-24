import torch
from torch import nn
import torch.nn.functional as F

import math
from inspect import isfunction


# 定义专家模型的最小容量
"""
MIN_EXPERT_CAPACITY: 专家模型的最小容量，设置为4。
用于在专家混合模型中，确保每个专家至少处理4个样本，以避免负载不均衡。
"""
MIN_EXPERT_CAPACITY = 4


def default(val, default_val):
    """
    如果 val 为 None，则返回 default_val；否则返回 val。
    如果 default_val 是一个函数，则先调用该函数获取默认值。

    参数:
        val: 输入值。
        default_val: 默认值，可以是一个值或一个函数。

    返回:
        如果 val 不为 None，则返回 val；否则返回 default_val 的值。
    """
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val


def cast_tuple(el):
    """
    将输入转换为元组。如果输入已经是元组，则直接返回；否则将其作为单个元素的元组返回。

    参数:
        el: 输入元素，可以是任意类型。

    返回:
        输入元素转换后的元组。
    """
    return el if isinstance(el, tuple) else (el,)


def top1(t):
    """
    获取张量 t 的每个样本的 top1 值及其索引。

    参数:
        t (torch.Tensor): 输入张量，形状为 (batch_size, ...)。

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 返回两个张量：
            - 第一个张量包含每个样本的 top1 值，形状为 (batch_size,)。
            - 第二个张量包含每个样本的 top1 值的索引，形状为 (batch_size,)。
    """
    # 对输入张量 t 在最后一个维度上执行 topk 操作，获取 top1 值和索引
    values, index = t.topk(k=1, dim=-1)
    # 使用 squeeze 去除最后一个维度，得到形状为 (batch_size,)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def cumsum_exclusive(t, dim=-1):
    """
    计算张量 t 在指定维度上的排他累计和（exclusive cumulative sum）。
    即，对于每个元素，其排他累计和为其前面所有元素的总和，不包括当前元素。

    参数:
        t (torch.Tensor): 输入张量。
        dim (int): 进行累计和的维度，默认为最后一个维度。

    返回:
        torch.Tensor: 输入张量在指定维度上的排他累计和。
    """
    # 获取输入张量的维度数
    num_dims = len(t.shape)
    # 计算需要填充的维度数量
    num_pad_dims = - dim - 1
    # 生成前填充参数，例如 (0, 0, 0, 0) 对于4维张量
    pre_padding = (0, 0) * num_pad_dims
    # 生成前切片参数，例如 (slice(None), slice(None), slice(None), slice(None))
    pre_slice   = (slice(None),) * num_pad_dims
    # 在指定维度前填充一个零，以便计算排他累计和
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    # 去除最后一个元素，得到排他累计和
    return padded_t[(..., slice(None, -1), *pre_slice)]


def safe_one_hot(indexes, max_length):
    """
    安全地生成 one-hot 编码，避免索引超出范围导致的错误。

    参数:
        indexes (torch.Tensor): 输入的索引张量。
        max_length (int): one-hot 编码的最大长度。

    返回:
        torch.Tensor: one-hot 编码后的张量。
    
    说明:
        如果输入索引的最大值大于 max_length，则截断超出部分以避免错误。
    """
    # 计算输入索引中的最大值，并加1得到 one-hot 编码的实际长度
    max_index = indexes.max() + 1
    # 生成 one-hot 编码，编码长度取 max_index + 1 和 max_length 中的较大值
    # 截取前 max_length 个维度，以避免超出范围的维度
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]


def init_(t):
    """
    使用均匀分布初始化张量，标准差为 1/sqrt(dim)，其中 dim 是张量的最后一个维度。

    参数:
        t (torch.Tensor): 需要初始化的张量。

    返回:
        torch.Tensor: 初始化后的张量。
    
    说明:
        初始化方法类似于 TensorFlow 中的默认初始化方法。
    """
    # 获取张量的最后一个维度
    dim = t.shape[-1]
    # 计算标准差
    std = 1 / math.sqrt(dim)
    # 使用均匀分布初始化张量，范围为 [-std, std]
    return t.uniform_(-std, std)


# activations

class GELU_(nn.Module):
    """
    高斯误差线性单元（GELU）激活函数的实现。
    如果 PyTorch 版本支持内置的 nn.GELU，则使用内置版本；否则使用自定义实现。

    前向传播:
        使用近似公式计算 GELU 激活函数:
            0.5 * x * (1 + tanh(math.sqrt(2/π) * (x + 0.044715 * x³)))
    """
    def forward(self, x):
        """
        前向传播方法，计算 GELU 激活函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 激活后的张量。
        """
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


# 如果 PyTorch 版本支持内置的 nn.GELU，则使用内置版本；否则使用自定义实现
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


# expert class

class Experts(nn.Module):
    """
    Experts 类实现了一个专家混合模型（Mixture of Experts, MoE）。
    该模型包含多个专家网络，每个专家由两层线性变换组成，中间使用激活函数。
    输入数据被分配给不同的专家进行处理，最终将所有专家的输出结合起来。

    参数说明:
        dim (int): 输入和输出的特征维度。
        num_experts (int 或 tuple, 可选): 专家的数量。如果传入的是 tuple，则每个专家使用不同的维度。
        hidden_dim (int, 可选): 隐藏层的维度。如果未指定，则默认为 dim * 4。
        activation (nn.Module, 可选): 激活函数，默认为 GELU。
    """
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU):
        super().__init__()

        # 如果未指定 hidden_dim，则默认为 dim * 4
        hidden_dim = default(hidden_dim, dim * 4)
        # 如果 num_experts 是 tuple，则直接使用；否则将其转换为 tuple
        num_experts = cast_tuple(num_experts)

        # 初始化第一层线性变换的权重，形状为 (num_experts, dim, hidden_dim)
        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        # 初始化第二层线性变换的权重，形状为 (num_experts, hidden_dim, dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        # 使用自定义的初始化方法初始化权重
        w1 = init_(w1)
        w2 = init_(w2)
        
        # 将权重注册为可训练的参数
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        # 实例化激活函数
        self.act = activation()

    def forward(self, x):
        """
        前向传播方法，对输入数据应用专家混合模型。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, ..., dim)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, ..., dim)。
        """
        # 使用 einsum 进行批量矩阵乘法，计算隐藏层的输出
        # '...nd,...dh->...nh' 表示对输入张量的最后两维进行矩阵乘法
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        # 应用激活函数
        hidden = self.act(hidden)
        # 使用 einsum 进行批量矩阵乘法，计算输出层的输出
        # '...nh,...hd->...nd' 表示对隐藏层输出张量的最后两维进行矩阵乘法
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out


# gating network

class Top2Gating(nn.Module):
    """
    Top2Gating 类实现了 Top-2 门控机制，用于专家混合模型（Mixture of Experts, MoE）。
    该机制为每个输入样本选择两个最合适的专家，并计算相应的门控权重。
    支持在训练和评估阶段使用不同的策略和阈值。

    参数说明:
        dim (int): 输入数据的特征维度。
        num_gates (int): 门控的数量，即专家的数量。
        eps (float, 可选): 用于数值稳定性的一个小常数，默认为 1e-9。
        outer_expert_dims (tuple, 可选): 外部专家维度，用于定义门控权重的形状，默认为空元组。
        second_policy_train (str, 可选): 训练阶段选择第二个专家的策略，默认为 'random'。
        second_policy_eval (str, 可选): 评估阶段选择第二个专家的策略，默认为 'random'。
        second_threshold_train (float, 可选): 训练阶段选择第二个专家的阈值，默认为 0.2。
        second_threshold_eval (float, 可选): 评估阶段选择第二个专家的阈值，默认为 0.2。
        capacity_factor_train (float, 可选): 训练阶段专家容量的乘数因子，默认为 1.25。
        capacity_factor_eval (float, 可选): 评估阶段专家容量的乘数因子，默认为 2.0。
    """
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        # 初始化门控权重，形状为 (*outer_expert_dims, dim, num_gates)
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        # 训练和评估阶段的不同策略和阈值
        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance = None):
        """
        前向传播方法，计算 Top-2 门控，并生成调度张量和组合张量。

        参数:
            x (torch.Tensor): 输入张量，形状为 (*, batch_size, group_size, dim)。
            importance (Optional[torch.Tensor], 可选): 重要性权重，形状为 (*, batch_size, group_size)。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 返回三个张量：
                - 第一个张量是调度张量，形状为 (batch_size, group_size, num_gates, expert_capacity)。
                - 第二个张量是组合张量，形状为 (batch_size, group_size, num_gates, expert_capacity)。
                - 第三个张量是损失张量，形状为 (1,)。
        """
        # 解包输入张量的形状
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates

        # 根据是否在训练阶段选择策略和阈值
        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        # 计算原始门控权重，使用 einsum 进行批量矩阵乘法
        raw_gates = torch.einsum('...bnd,...de->...bne', x, self.w_gating)
        # 对门控权重进行 softmax 归一化
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]
        # 寻找每个位置的前两个专家
        # 找到每个位置的第一个专家
        gate_1, index_1 = top1(raw_gates)
        # 创建第一个专家的独热编码掩码
        mask_1 = F.one_hot(index_1, num_gates).float()
        # 代理密度，用于后续计算
        density_1_proxy = raw_gates

        if importance is not None:
            # 如果提供了重要性权重，则根据重要性调整掩码和门控权重
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
            del equals_one_mask

        # 排除第一个专家，剩余的门控权重
        gates_without_top_1 = raw_gates * (1. - mask_1)

        # 找到每个位置的第二个专家
        gate_2, index_2 = top1(gates_without_top_1)
        # 创建第二个专家的独热编码掩码
        mask_2 = F.one_hot(index_2, num_gates).float()

        if importance is not None:
            # 如果提供了重要性权重，则根据重要性调整第二个专家的掩码
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
            del greater_zero_mask

        # normalize top2 gate scores
        # 归一化前两个门控分数
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # BALANCING LOSSES
        # shape = [batch, experts]
        # We want to equalize the fraction of the batch assigned to each expert
        # 平衡损失
        # 形状 = [batch, experts]
        # 我们希望平衡分配给每个专家的批次比例
        density_1 = mask_1.mean(dim=-2)
        # Something continuous that is correlated with what we want to equalize.
        # 与我们希望平衡的目标相关的连续性指标
        density_1_proxy = density_1_proxy.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        # Depending on the policy in the hparams, we may drop out some of the second-place experts.
        # 根据策略，可能丢弃一些第二名专家
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes
        # 每个序列最多向每个专家发送 expert_capacity 个位置
        # 需要静态的 expert_capacity 维度以适应专家的批量大小
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # COMPUTE ASSIGNMENT TO EXPERTS
        # [batch, group, experts]
        # This is the position within the expert's mini-batch for this sequence
        # 计算分配给专家的位置
        # [batch, group, experts]
        # 这是序列在这个专家的小批量中的位置
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        # Remove the elements that don't fit. [batch, group, experts]
        # 移除不适合的元素。[batch, group, experts]
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        # [batch, experts]
        # How many examples in this sequence go to this expert
        # [batch, experts]
        # 这个序列中有多少样本分配给这个专家
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        # [batch, group] - mostly ones, but zeros where something didn't fit
        # [batch, group] - 主要是1，但不适合的地方是0
        mask_1_flat = mask_1.sum(dim=-1)
        # [batch, group]
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)
        # Weight assigned to first expert.  [batch, group]
        # 分配给第一个专家的权重。[batch, group]
        gate_1 *= mask_1_flat

        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)

        position_in_expert_2 = position_in_expert_2.sum(dim=-1)
        gate_2 *= mask_2_flat
        
        # [batch, group, experts, expert_capacity]
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        return dispatch_tensor, combine_tensor, loss


# plain mixture of experts

class MoE(nn.Module):
    """
    MoE 类实现了专家混合模型（Mixture of Experts, MoE）。
    该模型通过 Top-2 门控机制选择两个最合适的专家处理输入数据，并结合专家的输出。
    支持在训练和评估阶段使用不同的策略和阈值，并可配置损失系数。

    参数说明:
        dim (int): 输入数据的特征维度。
        num_experts (int, 可选): 专家的数量，默认为16。
        hidden_dim (int, 可选): 专家网络中隐藏层的维度。如果未指定，则默认为 dim * 4。
        activation (nn.Module, 可选): 专家网络中使用的激活函数，默认为 nn.ReLU。
        second_policy_train (str, 可选): 训练阶段选择第二个专家的策略，默认为 'random'。
        second_policy_eval (str, 可选): 评估阶段选择第二个专家的策略，默认为 'random'。
        second_threshold_train (float, 可选): 训练阶段选择第二个专家的阈值，默认为 0.2。
        second_threshold_eval (float, 可选): 评估阶段选择第二个专家的阈值，默认为 0.2。
        capacity_factor_train (float, 可选): 训练阶段专家容量的乘数因子，默认为 1.25。
        capacity_factor_eval (float, 可选): 评估阶段专家容量的乘数因子，默认为 2.0。
        loss_coef (float, 可选): 损失系数，默认为 1e-2。
        experts (nn.Module, 可选): 自定义的专家网络。如果未指定，则使用默认的 Experts 类。
    """
    def __init__(self,
        dim,
        num_experts = 16,
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        # 专家数量
        self.num_experts = num_experts

        # 定义门控机制的关键字参数
        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        # 初始化 Top2Gating 门控机制
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        # 初始化专家网络，如果未提供自定义的 experts，则使用默认的 Experts 类
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        # 设置损失系数
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        """
        前向传播方法，执行专家混合模型的前向计算。

        参数:
            inputs (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。
            **kwargs: 其他关键字参数。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 返回两个张量：
                - 第一个张量是模型的输出，形状为 (batch_size, sequence_length, dim)。
                - 第二个张量是损失张量，形状为 (1,)。
        """
        # 获取输入张量的维度
        b, n, d, e = *inputs.shape, self.num_experts
        # 计算调度张量和组合张量，并计算门控损失
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        # 将输入数据根据调度张量分配给各个专家
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)

        # Now feed the expert inputs through the experts.
        # 调整专家输入的形状以适应专家网络
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        # 将专家输入通过专家网络
        expert_outputs = self.experts(expert_inputs)
        # 恢复专家输出的原始形状
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # 将专家输出根据组合张量进行组合，得到最终输出
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        # 返回最终输出和损失
        return output, loss * self.loss_coef


# 2-level heirarchical mixture of experts

class HeirarchicalMoE(nn.Module):
    """
    HeirarchicalMoE 类实现了双层专家混合模型（Hierarchical Mixture of Experts, HMoE）。
    该模型通过两层 Top-2 门控机制选择最合适的专家组和专家来处理输入数据，并结合专家的输出。
    支持在训练和评估阶段使用不同的策略和阈值，并可配置损失系数。

    参数说明:
        dim (int): 输入数据的特征维度。
        num_experts (tuple, 可选): 专家组的数量和每个专家组中专家的数量，默认为 (4, 4)。
        hidden_dim (int, 可选): 专家网络中隐藏层的维度。如果未指定，则默认为 dim * 4。
        activation (nn.Module, 可选): 专家网络中使用的激活函数，默认为 nn.ReLU。
        second_policy_train (str, 可选): 训练阶段选择第二个专家的策略，默认为 'random'。
        second_policy_eval (str, 可选): 评估阶段选择第二个专家的策略，默认为 'random'。
        second_threshold_train (float, 可选): 训练阶段选择第二个专家的阈值，默认为 0.2。
        second_threshold_eval (float, 可选): 评估阶段选择第二个专家的阈值，默认为 0.2。
        capacity_factor_train (float, 可选): 训练阶段专家容量的乘数因子，默认为 1.25。
        capacity_factor_eval (float, 可选): 评估阶段专家容量的乘数因子，默认为 2.0。
        loss_coef (float, 可选): 损失系数，默认为 1e-2。
        experts (nn.Module, 可选): 自定义的专家网络。如果未指定，则使用默认的 Experts 类。
    """
    def __init__(self,
        dim,
        num_experts = (4, 4),
        hidden_dim = None,
        activation = nn.ReLU,
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,
        experts = None):
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        # 外层专家的数量
        self.num_experts_outer = num_experts_outer
        # 内层专家的数量
        self.num_experts_inner = num_experts_inner

        # 定义门控机制的关键字参数
        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        # 初始化外层 Top2Gating 门控机制
        self.gate_outer = Top2Gating(dim, num_gates = num_experts_outer, **gating_kwargs)
        # 初始化内层 Top2Gating 门控机制
        self.gate_inner = Top2Gating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        # 初始化专家网络，如果未提供自定义的 experts，则使用默认的 Experts 类
        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        # 设置损失系数
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        """
        前向传播方法，执行双层专家混合模型的前向计算。

        参数:
            inputs (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, dim)。
            **kwargs: 其他关键字参数。

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 返回两个张量：
                - 第一个张量是模型的输出，形状为 (batch_size, sequence_length, dim)。
                - 第二个张量是损失张量，形状为 (1,)。
        """
        # 获取输入张量的维度
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        # 计算外层调度张量和组合张量，并计算外层门控损失
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        # 将输入数据根据外层调度张量分配给外层专家
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # we construct an "importance" Tensor for the inputs to the second-level
        # gating.  The importance of an input is 1.0 if it represents the
        # first-choice expert-group and 0.5 if it represents the second-choice expert
        # group.  This is used by the second-level gating.
        # 构建一个重要性张量，用于第二层门控。
        # 如果输入代表第一选择专家组，则重要性为1.0；如果代表第二选择专家组，则为0.5。
        # 这被第二层门控使用。
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        # 计算内层调度张量和组合张量，并计算内层门控损失
        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        # 将外层专家输入根据内层调度张量分配给内层专家
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # Now feed the expert inputs through the experts.
        # 将专家输入通过专家网络
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
        # expert_output has shape [y0, x1, h, d, n]
        # 组合专家输出（逆转之前的所有操作）
        # expert_output 的形状为 [y0, x1, h, d, n]

        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef
