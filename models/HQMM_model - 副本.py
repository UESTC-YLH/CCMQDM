import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import matrix_rank as rank
from constants import (Kraus_PERLAYES, LAYERS, batch_train,
                       batch_test, CLASS_task, Epochs, dim,
                       maxIter, tau, alpha, beta, n, class_num, d, epsilon, CLASS)
from tools.Tools import check_density_matrix


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
    output = []
    for i in range(CLASS):
        kraus_op = kraus_operators[-1][i]
        kraus_op = kraus_op.type(torch.complex64)
        evolved_state = torch.matmul(kraus_op, torch.matmul(mild_state, kraus_op.conj().t()))
        output.append(evolved_state)

    stacked_output = torch.stack(output)
    # print(stacked_output)
    return stacked_output


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
    povm_operators = []    # 生成POVM算子对应于测量结果为0的情况
    op_0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)
    povm_operators.append(op_0)    # 生成POVM算子对应于测量结果为1的情况
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
                 random_seed=42, class_num=CLASS_task, space="Complex",
                 kraus_operators=None, optimizer=None,
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
        self.n = n
        self.dim = dim
        self.tau = tau
        self.beta = beta
        self.space = space
        # self.alpha = alpha
        self.epochs = Epochs
        self.layers = layers
        # self.maxIter = maxIter
        self.class_num = class_num
        self.optimizer = optimizer
        self.random_seed = random_seed
        self.batch_num_test = batch_num_test
        self.povm_operators = povm_operators
        self.kraus_operators = kraus_operators
        self.dim_K = dim  # qubit
        self.matrix_size = 2
        self.goal_states = generate_orthogonal_unit_vectors(n)
        self.batch_num_train = batch_num_train
        self.k_number_per_layer = k_number_per_layer
        self.file_path_train_feature = file_path_train_feature
        self.file_path_train_label = file_path_train_label
        self.file_path_test_feature = file_path_test_feature
        self.file_path_test_label = file_path_test_label
        self.kraus_operators = []
        for layer in range(self.layers):
            kraus_op = self.generate_Kraus2(dim, Kraus_PERLAYES, epsilon)
            self.kraus_operators.append(nn.ParameterList([nn.Parameter(k.to(torch.complex64)) for k in kraus_op[0]]))
        self.k_measure_layers = self.generate_Kraus2(dim, CLASS, epsilon)
        self.kraus_operators.append(nn.ParameterList([nn.Parameter(k.to(torch.complex64)) for k in self.k_measure_layers[0]]))
        print("Kraus列表初始化完成")

    #     # 将所有参数合并到一个 ParameterList 中
    #     self.kraus_parameters = nn.ParameterList()
    #     for params in self.kraus_operators:
    #         self.kraus_parameters.extend(params)
    #     self.kraus_parameters.append(self.k_last_layer)
    #
    # def load_paras(self):
    #     self.k_measure_layers = self.generate_Kraus2(dim, CLASS, epsilon)
    #     self.k_last_layer = nn.Parameter(self.k_measure_layers[0].to(torch.complex64))
    #
    #     # 将所有参数合并到一个 ParameterList 中
    #     self.kraus_parameters = nn.ParameterList()
    #     for params in self.kraus_operators:
    #         self.kraus_parameters.extend(params)
    #     self.kraus_parameters.append(self.k_last_layer)

    def reshape_batch(self, data_list, batch_num):
        """
        根据batch大小来重新划分数据
        :param data_list: 原始数据列表
        :param batch_num: 每个batch的大小
        :return: 划分后的数据列表
        """
        num_matrices = len(data_list)
        num_batches = int(np.ceil(num_matrices / batch_num))
        batches = []
        for i in range(num_batches):
            start_idx = i * batch_num
            end_idx = min((i + 1) * batch_num, num_matrices)
            current_batch = data_list[start_idx:end_idx]
            batches.append(current_batch)
        return batches

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

    def generate_random_unitary(self, size):
        # 使用 QR 分解生成随机正交矩阵
        z = torch.randn(size, size, dtype=torch.complex64) + 1j * torch.randn(size, size, dtype=torch.complex64)
        q, r = torch.qr(z)
        d = torch.diag(r)
        ph = d / torch.abs(d)
        q = q * ph
        return q

    def generate_Kraus1(self, dim, Kraus_PERLAYES, epsilon):
        A = np.random.random([self.dim_K * Kraus_PERLAYES, self.dim_K * 2])
        ref = self.Linear_Dependent(A)
        B = np.transpose(A)
        if ref:
            print("Linear Dependent!")
            quit()
        else:
            # 对每一行进行施密特正交化,第一行为基础行，不进行正交
            for i in range(1, B.shape[0]):
                for j in range(0, i):
                    B[i, :] = B[i, :] - np.dot(B[i, :], B[j, :].T) / (np.linalg.norm(B[j, :], ord=2) ** 2) * B[j, :]

        C = np.zeros([self.dim_K * Kraus_PERLAYES, self.dim_K], dtype=complex).T
        for i in range(0, B.shape[0], 2):
            C[int(i / 2), :] = B[i, :] + 1j * B[i + 1, :]
            C[int(i / 2), :] = C[int(i / 2), :] / np.linalg.norm(C[int(i / 2), :], ord=2)
        Kraus_ops = C.T
        Kraus_ops = np.reshape(Kraus_ops, (Kraus_PERLAYES, 2, 2))
        Kraus_ops = torch.tensor(Kraus_ops, dtype=torch.complex128)
        self.check_kraus_condition(Kraus_ops)
        Kraus_ops_list = [Kraus_ops[i, :, :] for i in range(Kraus_ops.shape[0])]
        return Kraus_ops_list

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
        tol = 10e-6
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
            grad_ops = torch.cat(list(grads[self.k_number_per_layer * (layer): self.k_number_per_layer * (layer+1)]), dim=0)

            F = torch.norm(grad_ops, p=2)
            grad_ops = grad_ops / F
            grad_ops = self.beta * G_old + (1 - self.beta) * grad_ops
            E = torch.norm(grad_ops, p=2)
            grad_ops = grad_ops / E  # 二维

            U = torch.cat((grad_ops, kraus_ops), dim=1)  # 三维  12, 8?
            V = torch.cat((kraus_ops, -grad_ops), dim=1)  # 三维  12, 8?

            Inverse = torch.eye(2 * self.dim_K) + self.tau / 2 * torch.matmul(V.t().conj(), U)  # torch.Size([8, 8])
            item1 = torch.matmul(U, torch.inverse(Inverse))
            item2 = torch.matmul(V.t().conj(), kraus_ops)
            kraus_ops = torch.reshape(kraus_ops - self.tau * torch.matmul(item1, item2), (self.k_number_per_layer, dim, dim))
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

        print('last measure layers')
        layer = self.layers + 1
        measure_extracted_params = [param.data for param in self.kraus_operators[layer]]
        if not self.check_kraus_condition(torch.stack(measure_extracted_params)):
            raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')
        measure_kraus_ops = torch.cat([self.kraus_operators[layer][i] for i in range(self.k_number_per_layer)], dim=0)
        grad_ops = torch.cat(list(grads[self.k_number_per_layer * (layer): self.k_number_per_layer * (layer+1)]), dim=0)

        F1 = torch.norm(grad_ops, p=2)
        grad_ops = grad_ops / F1
        grad_ops = self.beta * G_old + (1 - self.beta) * grad_ops
        E1 = torch.norm(grad_ops, p=2)
        grad_ops = grad_ops / E1  # 二维

        U1 = torch.cat((grad_ops, measure_kraus_ops), dim=1)  # 三维  12, 8?
        V1 = torch.cat((measure_kraus_ops, -grad_ops), dim=1)  # 三维  12, 8?

        Inverse = torch.eye(2 * self.dim_K) + self.tau / 2 * torch.matmul(V1.t().conj(), U1)  # torch.Size([8, 8])
        item1 = torch.matmul(U1, torch.inverse(Inverse))
        item2 = torch.matmul(V1.t().conj(), measure_kraus_ops)
        measure_kraus_ops = torch.reshape(measure_kraus_ops - self.tau * torch.matmul(item1, item2), (self.k_number_per_layer, dim, dim))
        if measure_kraus_ops.shape != (self.k_number_per_layer, dim, dim):
            raise ValueError(
                f"Expected new_kraus_ops to have shape ({self.k_number_per_layer}, {dim}, {dim}), but got {measure_kraus_ops.shape}")
        # 更新特定层的 Kraus 操作符
        for i in range(self.k_number_per_layer):
            self.kraus_operators[-1][i].data = measure_kraus_ops[i]
        extracted_params = [param.data for param in self.kraus_operators[-1]]

        if not self.check_kraus_condition(torch.stack(extracted_params)):
            raise ValueError(r'不满足\sum_{m} K_{m}^{\dagger} K_{m}=I')
        check_list.append(extracted_params)

    # def measure_state(self, state):
    #     """
    #     对单个量子态进行测量,得到预测概率向量
    #     :param state: 输出的单个量子态密度矩阵
    #     :return: 预测概率向量,shape为(class_num,)
    #     """
    #     output_state = torch.zeros(self.class_num, dtype=torch.float32)
    #     povm_operators_complex = [op.to(torch.complex64) for op in self.povm_operators]
    #     for i in range(self.class_num):
    #         output_state[i] = torch.real(
    #             torch.trace(torch.matmul(povm_operators_complex[i], state)))
    #     output_state = output_state / torch.sum(output_state)
    #     return output_state

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
        prob_vector = []
        for idx, state in enumerate(input):
            state_tensor1 = torch.tensor(state.data, dtype=torch.complex64)
            state_tensor = self.encoder(state_tensor1, self.k_number_per_layer, self.kraus_operators)
            traces = torch.sum(state_tensor.diagonal(dim1=-2, dim2=-1), dim=-1).real
            state_tensors.append(state_tensor)
            prob_vector.append(traces)
        predicted_probs_tensor = torch.stack(prob_vector)
        return predicted_probs_tensor

