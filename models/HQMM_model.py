import os
import csv
import pickle
import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import matrix_rank as rank
from constants import (Kraus_PERLAYES, LAYERS, batch_train, p0, pT,
                       batch_test, CLASS_task, Epochs, dim,
                       pre_layers, tau_perlayer, tau_train_all, pre_Kraus_PERLAYES, beta, n, class_num, filepath, epsilon)
from tools.Tools import check_density_matrix, save_kraus, load_kraus, fidelity_fun


def encoder(state_tensor, k_number_per_layer, kraus_operators):
    """
    :param state_tensor:
    :param k_number_per_layer:
    :param kraus_operators:
    :return:
    """
    mild_state = state_tensor
    for layer in range(LAYERS):
        output_state = torch.zeros_like(state_tensor)
        for i in range(k_number_per_layer):
            kraus_op = kraus_operators[layer][i]
            kraus_op = kraus_op.type(torch.complex64)
            evolved_state = torch.matmul(kraus_op, torch.matmul(mild_state, kraus_op.conj().t()))
            output_state += evolved_state
        mild_state = output_state
    return mild_state


def encoder_all(state_tensor, k_number_per_layer, kraus_operators):
    """
    :param state_tensor:
    :param k_number_per_layer:
    :param kraus_operators:
    :return:
    """
    mild_state = state_tensor
    for layer in range(LAYERS*pre_layers):
        output_state = torch.zeros_like(state_tensor)
        for i in range(k_number_per_layer):
            kraus_op = kraus_operators[layer][i]
            kraus_op = kraus_op.type(torch.complex64)
            evolved_state = torch.matmul(kraus_op, torch.matmul(mild_state, kraus_op.conj().t()))
            output_state += evolved_state
        mild_state = output_state
    return mild_state


def encoder_all_constrained(state_tensor, k_number_per_layer, kraus_operators):
    """
    :param state_tensor:
    :param k_number_per_layer:
    :param kraus_operators:
    :return:
    """
    mild_state = state_tensor
    for layer in range(LAYERS):
        output_state = torch.zeros_like(state_tensor)
        for i in range(k_number_per_layer):
            kraus_op = kraus_operators[layer][i]
            kraus_op = kraus_op.type(torch.complex64)
            evolved_state = torch.matmul(kraus_op, torch.matmul(mild_state, kraus_op.conj().t()))
            output_state += evolved_state
        mild_state = output_state
    return mild_state


def encoder_pre(state_tensor, kraus_operators):
    """
    :param state_tensor:
    :param k_number_per_layer:
    :param kraus_operators:
    :return:
    """
    mild_state = state_tensor
    for layer in range(len(kraus_operators)):
        output_state = torch.zeros_like(state_tensor)
        for i in range(kraus_operators[0][0].shape[0]):
            kraus_op = kraus_operators[layer][0][i]
            kraus_op = kraus_op.type(torch.complex64)
            evolved_state = torch.matmul(kraus_op, torch.matmul(mild_state, kraus_op.conj().t()))
            output_state += evolved_state
        mild_state = output_state
    # fidelity_fun(mild_state, state_tensor)
    return mild_state


def generate_povm():
    """
    生成POVM测量算子
    :param class_num: 类别数量
    :return: POVM测量算子列表
    """
    povm_operators = []

    for i in range(class_num):
        random_matrix = torch.rand(2, 2)  # 随机生成一个2x2的张量
        positive_operator = torch.matmul(random_matrix, random_matrix.t())  # 计算正定算子
        povm_operators.append(positive_operator)

    sum_operator = sum(povm_operators)
    for i in range(class_num):
        povm_operators[i] /= sum_operator  # 归一化
    return povm_operators


def generate_01_povm():
    """
    生成01分类任务的POVM算子
    :return: 两个POVM算子列表，分别对应于测量结果为0和1的情况
    """
    povm_operators = []  # 生成POVM算子对应于测量结果为0的情况
    op_0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)
    povm_operators.append(op_0)  # 生成POVM算子对应于测量结果为1的情况
    op_1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64)
    povm_operators.append(op_1)
    return povm_operators


povm_operators = generate_01_povm()


def generate_unit_vector(n):
    vector = np.random.randn(n) + 1j * np.random.randn(n)
    unit_vector = vector / np.linalg.norm(vector)
    return unit_vector


def generate_orthogonal_unit_vectors(n):
    goal_states = []
    u = generate_unit_vector(n)
    v = generate_unit_vector(n)
    # 使用 Gram-Schmidt 正交化方法使 v 与 u 正交
    v = v - np.dot(u.conjugate(), v) * u
    v = v / np.linalg.norm(v)  # 单位化
    goal_states.append(torch.tensor(u))
    goal_states.append(torch.tensor(v))
    return goal_states


class quantum_hidden_Markov_neural_network(nn.Module):
    def __init__(self, k_number_per_layer=Kraus_PERLAYES, layers=LAYERS,
                 batch_num_train=batch_train, batch_num_test=batch_test,
                 class_num=CLASS_task, space="Complex", optimizer=None,
                 file_path_train_feature='data/Encode/Encode_train_feature.csv',
                 file_path_train_label='data/Encode/Encode_train_label.csv',
                 file_path_test_feature='data/Encode/Encode_test_feature.csv',
                 file_path_test_label='data/Encode/Encode_test_label.csv'):
        """
        量子隐马尔可夫神经网络初始化
        :param k_number_per_layer: 神经网络每一层分裂的K算子个数  5
        :param layers: 神经网络层数  5
        :param batch_num_train: 训练数据的batch个数,所有数据分为几批训练完
        :param batch_num_test: 测试数据的batch个数
        :param random_seed: 初始化种子
        :param class_num: 分类任务的分类个数
        :param space: K算符类型,为实数还是复数
        """
        super(quantum_hidden_Markov_neural_network, self).__init__()
        self.encoder = encoder
        self.encoder_all = encoder_all
        self.encoder_pre = encoder_pre
        self.n = n
        self.dim = dim
        self.tau = tau_perlayer
        self.beta = beta
        self.space = space
        self.epochs = Epochs
        self.layers = layers
        self.class_num = class_num
        self.optimizer = optimizer
        self.batch_num_test = batch_num_test
        self.povm_operators = povm_operators
        self.goal_states = generate_orthogonal_unit_vectors(n)
        self.batch_num_train = batch_num_train
        self.k_number_per_layer = k_number_per_layer
        self.file_path_train_feature = file_path_train_feature
        self.file_path_train_label = file_path_train_label
        self.file_path_test_feature = file_path_test_feature
        self.file_path_test_label = file_path_test_label
        self.kraus_operators = []
        kraus_ops = []
        self.kraus_ops_pre = []

        for layer in range(self.layers):
            # 可训练的模型参数，用于解躁
            kraus_op = self.generate_Kraus2(dim, Kraus_PERLAYES, epsilon)  # d
            kraus_ops.append(kraus_op)
            self.kraus_operators.append(
                nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64)) for k in kraus_op[0]]))
        save_kraus(kraus_ops, fr'{filepath}/train_initial_kraus_list.csv')

        T = pre_layers  # 总时间步数

        # 计算每个时间步的噪声强度
        time_steps = np.arange(T + 1)  # 时间步 0 到 T
        pt = p0 + (pT - p0) * time_steps / T  # 线性增加的噪声强度
        for p in pt:
            kraus_op_pre = self.generate_Depolarizing_Kraus(p)
            self.kraus_ops_pre.append(kraus_op_pre)
        save_kraus(self.kraus_ops_pre, fr'{filepath}/initial_kraus_list.csv')

        # 将所有参数合并到一个 ParameterList中
        self.kraus_parameters = nn.ParameterList()
        for params in self.kraus_operators:
            self.kraus_parameters.extend(params)

    @staticmethod
    def Linear_Dependent(M):
        """
        判断一个矩阵中的行向量是否线性相关
        :param M: 矩阵
        :return: 布尔值，True:线性相关， False:线性无关
        """
        if rank(M) < min(M.shape):
            return True
        else:
            return False

    def generate_Depolarizing_Kraus(self, p):
        # 计算系统的维度 d = 2^n
        d = 2 ** self.n
        all_kraus_operators = []
        # 定义单量子比特的去极化噪声 Kraus 矩阵
        E_0 = np.sqrt(1 - p) * np.array([[1, 0], [0, 1]])  # 单位矩阵
        E_1 = np.sqrt(p) * np.array([[0, 1], [1, 0]])  # Pauli-X 操作

        # 定义返回的 Kraus 矩阵列表
        kraus_matrices = []

        # 使用张量积生成多比特的 Kraus 矩阵
        for i in range(d):  # 共 2^n 种可能的比特状态
            kraus = 1
            for j in range(self.n):  # 对每个量子比特选择 E_0 或 E_1
                if (i >> j) & 1:  # 如果该比特是1，使用 E_1
                    kraus = np.kron(kraus, E_1)
                else:  # 如果该比特是0，使用 E_0
                    kraus = np.kron(kraus, E_0)
            kraus_matrices.append(kraus)
        # print("kraus_matrices", kraus_matrices)
        all_kraus_operators.append(kraus_matrices)
        return torch.tensor(all_kraus_operators)

    def generate_Kraus2(self, d, Kraus_PERLAYES, epsilon):
        n = Kraus_PERLAYES
        all_kraus_operators = []

        # 初始设置
        p = np.random.rand(n)
        p /= np.sum(p)
        identity = np.eye(d)

        # 生成初始的 Kraus 算子
        K = []
        for i in range(n):
            A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
            K.append(np.sqrt(p[i]) * identity + epsilon * A)

        # 计算 \Lambda
        Lambda = sum(K[i].conj().T @ K[i] for i in range(n))

        # 计算 \Lambda 的逆平方根
        eigvals, eigvecs = np.linalg.eigh(Lambda)
        Lambda_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T

        # 调整 Kraus 算子
        K_tilde = [K[i] @ Lambda_inv_sqrt for i in range(n)]

        # 验证 \sum K_m^\dagger K_m = I
        I = sum(K_tilde[i].conj().T @ K_tilde[i] for i in range(n))
        if not np.allclose(I, np.eye(d)):
            raise ValueError(f'Layer: \sum K_m^\dagger K_m != I')

        all_kraus_operators.append(K_tilde)
        return torch.tensor(all_kraus_operators)

    def check_kraus_condition(self, kraus_operators):
        tol = 10e-5
        kraus_product_sum = torch.zeros_like(kraus_operators[0])
        kraus_product_sum = torch.tensor(kraus_product_sum.clone().detach(), dtype=torch.complex128)
        for kraus_operator in kraus_operators:
            kraus_conjugate = torch.conj(kraus_operator.transpose(0, 1))
            kraus_product = torch.matmul(kraus_conjugate, kraus_operator)
            kraus_product_sum += kraus_product
        identity_matrix = torch.eye(kraus_operators.shape[-1], dtype=torch.complex128)
        if torch.allclose(kraus_product_sum, identity_matrix, atol=tol):
            return True
        else:
            raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')

    def update_kraus_operators(self, grads):
        """
        更新Kraus算子
        :param grads: 损失函数对Kraus算子的梯度$$ G = \frac { \partial L } { \partial k }$$
        $$ K = K - \tau U ( I + \frac { \tau } { 2 } V ^ { + } U ) ^ { - 1 } V ^ { + } k$$
        """
        check_list = []
        # print('self.k_last_layer', self.k_last_layer)
        G_old = 0
        for layer in range(self.layers):
            extracted_params = [param.data for param in self.kraus_operators[layer]]
            if not self.check_kraus_condition(torch.stack(extracted_params)):
                raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')
            kraus_ops = torch.cat([self.kraus_operators[layer][i] for i in range(self.k_number_per_layer)], dim=0)
            grad_ops = torch.cat(list(grads[self.k_number_per_layer * (layer): self.k_number_per_layer * (layer + 1)]),
                                 dim=0)

            F = torch.norm(grad_ops, p=2)
            grad_ops = grad_ops / F
            grad_ops = self.beta * G_old + (1 - self.beta) * grad_ops
            E = torch.norm(grad_ops, p=2)
            grad_ops = grad_ops / E  # 二维

            U = torch.cat((grad_ops, kraus_ops), dim=1)  # 三维  12, 8?
            V = torch.cat((kraus_ops, -grad_ops), dim=1)  # 三维  12, 8?

            Inverse = torch.eye(2 * self.dim) + self.tau / 2 * torch.matmul(V.t().conj(), U)  # torch.Size([8, 8])
            item1 = torch.matmul(U, torch.inverse(Inverse))
            item2 = torch.matmul(V.t().conj(), kraus_ops)
            kraus_ops = torch.reshape(kraus_ops - self.tau * torch.matmul(item1, item2),
                                      (self.k_number_per_layer, dim, dim))
            if kraus_ops.shape != (self.k_number_per_layer, dim, dim):
                raise ValueError(
                    f"Expected new_kraus_ops to have shape ({self.k_number_per_layer}, {dim}, {dim}), but got {kraus_ops.shape}")
            # 更新特定层的 Kraus 操作符
            for i in range(self.k_number_per_layer):
                self.kraus_operators[layer][i].data = kraus_ops[i]
            extracted_params = [param.data for param in self.kraus_operators[layer]]

            if not self.check_kraus_condition(torch.stack(extracted_params)):
                raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')
            G_old = grad_ops
            check_list.append(extracted_params)

    def measure_state(self, state):
        """
        对量子态进行测量，计算测量算子下的测量概率。

        参数:
        - state: 量子态的密度矩阵 (torch.Tensor)，应具有 requires_grad=True。

        返回:
        - pred_probs_tensor: 每个测量算子下的测量结果概率 (torch.Tensor)，梯度信息将被保留。
        """
        # 确保 state 张量参与梯度计算
        if not state.requires_grad:
            state = state.detach().requires_grad_()

        # 确保密度矩阵是Hermitian的
        rho = (state + state.conj().transpose(-1, -2)) / 2

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(rho, UPLO='L')

        # 特征值需要为非负，因为它们代表概率
        eigenvalues = torch.clamp_min(eigenvalues.real, 0)

        # 特征值的实部表示测量的概率
        probabilities = eigenvalues

        result1 = probabilities[0] + probabilities[1]
        result2 = probabilities[2] + probabilities[3]

        # 使用 torch.stack 将结果拼接成一个张量
        result = torch.stack([result1, result2])
        return result

    def forward_pre(self, states, l):
        state_tensors = []
        for state in states:
            state_tensor1 = torch.tensor(state, dtype=torch.complex64)
            state_tensor = self.encoder_pre(state_tensor1, self.kraus_ops_pre[:l])
            state_tensors.append(state_tensor)
        return state_tensors

    def forward(self, input):
        """
        $$ \rho ( t + \Delta t ) = \sum _ { m } K _ { m } \rho ( t ) K _ { m } ^ { + }$$
        假设观测结果为y:
            $$ \rho ( t + \Delta t ) = \frac { K _ { y } \rho ( t ) K _ { y } ^ { + } }
            { T r \left[ K _ { y } \rho ( t ) K _ { y } ^ { + } \right] }$$
        $$ 观 测 到 结 果 为 y 的 概 率 为 :
            P ( y ，t ) = T r \left[ K _ { y } \rho ( t ) K ^ { + } _ { y } \right] 。$$
        前向传播,得到\rho_out
        :return: 预测概率矩阵,shape为(batch_size, class_num)
        """
        state_tensors = []
        for idx, state in enumerate(input):
            state_tensor1 = torch.tensor(state, dtype=torch.complex64, requires_grad=True)
            state_tensor = self.encoder(state_tensor1, self.k_number_per_layer, self.kraus_operators)
            # fidelity += state_fidelity(state_tensor1.detach().numpy(), state_tensor.detach().numpy())
            state_tensors.append(state_tensor)
        return state_tensors


def check_kraus_condition(kraus_operators):
    tol = 10e-5
    kraus_product_sum = torch.zeros_like(kraus_operators[0])
    kraus_product_sum = torch.tensor(kraus_product_sum.clone().detach(), dtype=torch.complex128)
    for kraus_operator in kraus_operators:
        kraus_conjugate = torch.conj(kraus_operator.transpose(0, 1))
        kraus_product = torch.matmul(kraus_conjugate, kraus_operator)
        kraus_product_sum += kraus_product
    identity_matrix = torch.eye(kraus_operators.shape[-1], dtype=torch.complex128)
    if torch.allclose(kraus_product_sum, identity_matrix, atol=tol):
        return True
    else:
        raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')


class QHMM_train_all(nn.Module):
    def __init__(self, k_number_per_layer=Kraus_PERLAYES, layers=LAYERS):

        super(QHMM_train_all, self).__init__()

        self.encoder_all = encoder_all
        self.encoder_pre = encoder_pre
        self.n = n
        self.dim = dim
        self.tau = tau_train_all
        self.beta = beta
        self.layers = layers
        self.kraus_ops_pre = []
        self.kraus_operators_all = []
        self.kraus_parameters_all = []
        self.k_number_per_layer = k_number_per_layer

        T = pre_layers  # 总时间步数

        # 计算每个时间步的噪声强度
        time_steps = np.arange(T + 1)  # 时间步 0 到 T
        pt = p0 + (pT - p0) * time_steps / T  # 线性增加的噪声强度
        for p in pt:
            kraus_op_pre = self.generate_Depolarizing_Kraus(p)
            self.kraus_ops_pre.append(kraus_op_pre)
        save_kraus(self.kraus_ops_pre, fr'{filepath}/initial_kraus_list.csv')

        for K in range(pre_layers, 0, -1):
            kraus_op = load_kraus(fr'{filepath}/kraus_operators_array_{K}.csv')  # d
            for l in range(LAYERS):
                self.kraus_operators_all.append(nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64))
                                                                  for k in kraus_op[l * Kraus_PERLAYES:
                                                                                    (l + 1) * Kraus_PERLAYES]]))

        # for l in range(pre_layers, 0, -1):
        #     kraus_op = load_kraus(fr'{filepath}/kraus_operators_array_{l}.csv')
        #     for l in range(LAYERS):
        #         self.kraus_operators.append(nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64)) for k
        #                                                       in
        #                                                       kraus_op[l * Kraus_PERLAYES:(l + 1) * Kraus_PERLAYES]]))
        #
        self.kraus_parameters_all = nn.ParameterList()
        for params in self.kraus_operators_all:
            self.kraus_parameters_all.extend(params)
        print("self.kraus_operators_all", self.kraus_operators_all)

    def generate_Depolarizing_Kraus(self, p):
        # 计算系统的维度 d = 2^n
        d = 2 ** self.n
        all_kraus_operators = []
        # 定义单量子比特的去极化噪声 Kraus 矩阵
        E_0 = np.sqrt(1 - p) * np.array([[1, 0], [0, 1]])  # 单位矩阵
        E_1 = np.sqrt(p) * np.array([[0, 1], [1, 0]])  # Pauli-X 操作

        # 定义返回的 Kraus 矩阵列表
        kraus_matrices = []

        # 使用张量积生成多比特的 Kraus 矩阵
        for i in range(d):  # 共 2^n 种可能的比特状态
            kraus = 1
            for j in range(self.n):  # 对每个量子比特选择 E_0 或 E_1
                if (i >> j) & 1:  # 如果该比特是1，使用 E_1
                    kraus = np.kron(kraus, E_1)
                else:  # 如果该比特是0，使用 E_0
                    kraus = np.kron(kraus, E_0)
            kraus_matrices.append(kraus)
        # print("kraus_matrices", kraus_matrices)
        all_kraus_operators.append(kraus_matrices)
        return torch.tensor(all_kraus_operators)

    def forward_all(self, input):
        state_tensors = []
        for idx, state in enumerate(input):
            state_tensor1 = torch.tensor(state, dtype=torch.complex64, requires_grad=True)
            state_tensor = self.encoder_all(state_tensor1, self.k_number_per_layer, self.kraus_operators_all)
            state_tensors.append(state_tensor)
        return state_tensors

    def update_kraus_operators_all(self, grads):
        """
        更新Kraus算子
        :param grads: 损失函数对Kraus算子的梯度$$ G = \frac { \partial L } { \partial k }$$
        $$ K = K - \tau U ( I + \frac { \tau } { 2 } V ^ { + } U ) ^ { - 1 } V ^ { + } k$$
        """
        check_list = []
        # print('self.k_last_layer', self.k_last_layer)
        G_old = 0
        for layer in range(self.layers*pre_layers):
            extracted_params = [param.data for param in self.kraus_operators_all[layer]]
            if not check_kraus_condition(torch.stack(extracted_params)):
                raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')
            kraus_ops = torch.cat([self.kraus_operators_all[layer][i] for i in range(self.k_number_per_layer)], dim=0)
            grad_ops = torch.cat(list(grads[self.k_number_per_layer * (layer): self.k_number_per_layer * (layer + 1)]),
                                 dim=0)

            F = torch.norm(grad_ops, p=2)
            grad_ops = grad_ops / F
            grad_ops = self.beta * G_old + (1 - self.beta) * grad_ops
            E = torch.norm(grad_ops, p=2)
            grad_ops = grad_ops / E  # 二维

            U = torch.cat((grad_ops, kraus_ops), dim=1)  # 三维  12, 8?
            V = torch.cat((kraus_ops, -grad_ops), dim=1)  # 三维  12, 8?

            Inverse = torch.eye(2 * self.dim) + self.tau / 2 * torch.matmul(V.t().conj(), U)  # torch.Size([8, 8])
            item1 = torch.matmul(U, torch.inverse(Inverse))
            item2 = torch.matmul(V.t().conj(), kraus_ops)
            kraus_ops = torch.reshape(kraus_ops - self.tau * torch.matmul(item1, item2),
                                      (self.k_number_per_layer, dim, dim))
            if kraus_ops.shape != (self.k_number_per_layer, dim, dim):
                raise ValueError(
                    f"Expected new_kraus_ops to have shape ({self.k_number_per_layer}, {dim}, {dim}), but got {kraus_ops.shape}")
            # 更新特定层的 Kraus 操作符
            for i in range(self.k_number_per_layer):
                self.kraus_operators_all[layer][i].data = kraus_ops[i]
            extracted_params = [param.data for param in self.kraus_operators_all[layer]]

            if not check_kraus_condition(torch.stack(extracted_params)):
                raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')
            G_old = grad_ops
            check_list.append(extracted_params)

    def forword_all(self, states):
        state_tensors = []
        for state in states:
            state_tensor1 = torch.tensor(state, dtype=torch.complex64)
            state_tensor = self.encoder_pre(state_tensor1, self.kraus_ops_pre_all)
            state_tensors.append(state_tensor)
        return state_tensors


class QHMM_train_all_constrained(nn.Module):
    def __init__(self):

        super(QHMM_train_all_constrained, self).__init__()
        self.encoder_all_constrained = encoder_all_constrained
        self.encoder_pre = encoder_pre
        self.n = n
        self.dim = dim
        self.tau = tau_train_all
        self.beta = beta
        self.layers = LAYERS
        self.kraus_ops_pre = []
        self.kraus_operators_all = []
        self.kraus_parameters_all = []
        self.k_number_per_layer = Kraus_PERLAYES
        kraus_op = load_kraus(fr'{filepath}/kraus_operators_array_{pre_layers}.csv')  # d

        for K in range(pre_layers, 0, -1):
            # kraus_op = load_kraus(fr'{filepath}/kraus_operators_array_{K}.csv')  # d
            for l in range(LAYERS):
                self.kraus_operators_all.append(nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64))
                                                                  for k in kraus_op[l * Kraus_PERLAYES:
                                                                                    (l + 1) * Kraus_PERLAYES]]))

        self.kraus_parameters_all = nn.ParameterList()
        for params in self.kraus_operators_all:
            self.kraus_parameters_all.extend(params)

    def forward_all(self, input):
        all_state_tensors = []
        for idx, kraus in enumerate(input):
            state_tensors = []
            kraus_operators_all = self.kraus_operators_all[self.layers * idx: (idx + 1) * self.layers]
            for idxs, state in enumerate(kraus):
                state_tensor = self.encoder_all_constrained(state, self.k_number_per_layer, kraus_operators_all)
                state_tensors.append(state_tensor)
            all_state_tensors.append(state_tensors)
        return all_state_tensors

    def update_kraus_operators_all(self, grads):
        """
        更新Kraus算子
        :param grads: 损失函数对Kraus算子的梯度$$ G = \frac { \partial L } { \partial k }$$
        $$ K = K - \tau U ( I + \frac { \tau } { 2 } V ^ { + } U ) ^ { - 1 } V ^ { + } k$$
        """
        check_list = []
        G_old = 0
        for layer in range(self.layers * pre_layers):
            extracted_params = [param.data for param in self.kraus_operators_all[layer]]
            if not check_kraus_condition(torch.stack(extracted_params)):
                raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')
            kraus_ops = torch.cat([self.kraus_operators_all[layer][i] for i in range(self.k_number_per_layer)], dim=0)
            grad_ops = torch.cat(list(grads[self.k_number_per_layer * (layer): self.k_number_per_layer * (layer + 1)]),
                                 dim=0)

            F = torch.norm(grad_ops, p=2)
            grad_ops = grad_ops / F
            grad_ops = self.beta * G_old + (1 - self.beta) * grad_ops
            E = torch.norm(grad_ops, p=2)
            grad_ops = grad_ops / E  # 二维

            U = torch.cat((grad_ops, kraus_ops), dim=1)  # 三维  12, 8?
            V = torch.cat((kraus_ops, -grad_ops), dim=1)  # 三维  12, 8?

            Inverse = torch.eye(2 * self.dim) + self.tau / 2 * torch.matmul(V.t().conj(), U)  # torch.Size([8, 8])
            item1 = torch.matmul(U, torch.inverse(Inverse))
            item2 = torch.matmul(V.t().conj(), kraus_ops)
            kraus_ops = torch.reshape(kraus_ops - self.tau * torch.matmul(item1, item2),
                                      (self.k_number_per_layer, dim, dim))
            if kraus_ops.shape != (self.k_number_per_layer, dim, dim):
                raise ValueError(
                    f"Expected new_kraus_ops to have shape ({self.k_number_per_layer}, {dim}, {dim}), but got {kraus_ops.shape}")
            # 更新特定层的 Kraus 操作符
            for i in range(self.k_number_per_layer):
                self.kraus_operators_all[layer][i].data = kraus_ops[i]
            extracted_params = [param.data for param in self.kraus_operators_all[layer]]

            if not check_kraus_condition(torch.stack(extracted_params)):
                raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')
            G_old = grad_ops
            check_list.append(extracted_params)
