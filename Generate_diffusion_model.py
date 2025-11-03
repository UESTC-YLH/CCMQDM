import torch
import torch.nn as nn
from constants import (LAYERS, pre_layers, Kraus_PERLAYES, filepath)
from models import HQMM_model
from tools.Tools import load_kraus


class Generating_diffusion_model(nn.Module):
    def __init__(self):
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
        super(Generating_diffusion_model, self).__init__()
        self.encoder = HQMM_model.encoder
        self.kraus_operators = []
        self.k_number_per_layer = Kraus_PERLAYES
        # for l in range(1, pre_layers + 1):
        for l in range(pre_layers, 0, -1):
            kraus_op = load_kraus(fr'{filepath}/kraus_operators_array_{l}.csv')
            for l in range(LAYERS):
                self.kraus_operators.append(nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64)) for k
                                                              in
                                                              kraus_op[l * Kraus_PERLAYES:(l + 1) * Kraus_PERLAYES]]))
                # self.kraus_operators.append(nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64)) for k
                #                                               in kraus_op]))

        self.kraus_parameters = nn.ParameterList()
        for params in self.kraus_operators:
            self.kraus_parameters.extend(params)
        # print("self.kraus_operators", self.kraus_operators)
        # print("self.kraus_parameters", self.kraus_parameters)

    def generation_forward(self, input):
        state_tensors = []
        for idx, state in enumerate(input):
            state_tensor1 = torch.tensor(state, dtype=torch.complex64, requires_grad=False)
            state_tensor = self.encoder(state_tensor1, self.k_number_per_layer, self.kraus_operators)
            state_tensors.append(state_tensor)
        return state_tensors


class Generating_diffusion_model_all(nn.Module):
    def __init__(self):
        super(Generating_diffusion_model_all, self).__init__()
        self.encoder = HQMM_model.encoder
        self.kraus_operators = []
        self.k_number_per_layer = Kraus_PERLAYES
        kraus_op = load_kraus(fr'{filepath}/kraus_operators_array_0.csv')  # d
        for l in range(LAYERS):
            self.kraus_operators.append(nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64)) for k
                                                          in
                                                          kraus_op[l * Kraus_PERLAYES:(l + 1) * Kraus_PERLAYES]]))

            # self.kraus_operators.append(nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64)) for k
            #                                               in kraus_op]))
        self.kraus_parameters = nn.ParameterList()
        for params in self.kraus_operators:
            self.kraus_parameters.extend(params)
        # print("self.kraus_operators", self.kraus_operators)
        # print("self.kraus_parameters", self.kraus_parameters)

    def generation_forward(self, input):
        state_tensors = []
        for idx, state in enumerate(input):
            state_tensor1 = torch.tensor(state, dtype=torch.complex64, requires_grad=False)
            state_tensor = self.encoder(state_tensor1, self.k_number_per_layer, self.kraus_operators)
            state_tensors.append(state_tensor)
        return state_tensors


class Generating_diffusion_model_all_constrained(nn.Module):
    def __init__(self):
        super(Generating_diffusion_model_all_constrained, self).__init__()
        self.encoder = HQMM_model.encoder_all_constrained
        self.kraus_operators = []
        self.k_number_per_layer = Kraus_PERLAYES
        kraus_ops = load_kraus(fr'{filepath}/kraus_operators_array_0_constrained.csv')  # d
        for K in range(pre_layers):
            kraus_op = kraus_ops[K * Kraus_PERLAYES * LAYERS:(K + 1) * Kraus_PERLAYES * LAYERS]
            for l in range(LAYERS):
                self.kraus_operators.append(nn.ParameterList([nn.Parameter(torch.tensor(k).to(torch.complex64)) for k
                                                              in
                                                              kraus_op[l * Kraus_PERLAYES:(l + 1) * Kraus_PERLAYES]]))

        self.kraus_parameters = nn.ParameterList()
        for params in self.kraus_operators:
            self.kraus_parameters.extend(params)

    def generation_forward(self, input):
        all_state_tensors = []
        for idx, kraus in enumerate(input):
            state_tensors = []
            kraus_operators_all = self.kraus_operators[LAYERS * idx: (idx + pre_layers) * LAYERS]
            for idxs, state in enumerate(kraus):
                state_tensor = self.encoder(state, self.k_number_per_layer, kraus_operators_all)
                state_tensors.append(state_tensor)
            all_state_tensors.append(state_tensors)
        return all_state_tensors
