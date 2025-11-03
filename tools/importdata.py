import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.quantum_info import partial_trace
from qiskit_aer import Aer
from tools.Tools import write_density_matrices, write_label_to_csv, check_matrix_conditions, sum_normalize_density_matrices


class read_encode_dateset:

    def __init__(self, CLASS=10, encoding_method='amplitude_code', train_num=5000, test_num=100,
                 file_path_train=fr'D:\pycharm_projects\2312_QHMNN\data\Encode/Encode_train_feature.csv',
                 file_path_train_label=fr'D:\pycharm_projects\2312_QHMNN\data\Encode/Encode_train_label.csv',
                 file_path_test=fr'D:\pycharm_projects\2312_QHMNN\data\Encode/Encode_test_feature.csv',
                 file_path_test_label=fr'D:\pycharm_projects\2312_QHMNN\data\Encode/Encode_test_label.csv',
                 ):
        """
        :param CLASS: 分类个数
        :param encoding_method: 编码方式
        :param train_num: 训练集个数
        :param test_num: 测试集个数
        :param file_path_train: 训练集编码输出位置
        :param file_path_train_label: 训练集编码标签输出位置
        :param file_path_test: 测试集编码输出位置
        :param file_path_test_label: 测试集编码标签输出位置
        """
        self.CLASS = CLASS
        self.train_num = train_num
        self.test_num = test_num
        self.file_path_train = file_path_train
        self.file_path_test = file_path_test
        self.file_path_train_label = file_path_train_label
        self.file_path_test_label = file_path_test_label
        self.encoding_method = encoding_method
        # 数据集读取结果默认存储在data中，train和test的编码结果分别存在两个csv中，每次运行记得清空data中数据

    def dataset_load(self, mode):  # 加载数据
        trans = ToTensor()
        mnist_train = []
        mnist_test = []
        # 分别读取训练集和测试集
        if mode == 'train':
            mnist_train = FashionMNIST(root="./data", train=True, transform=trans, download=True)
            total_length = len(mnist_train)
            random_indices = []
            for i in range(self.train_num):
                while True:
                    random_index = np.random.randint(total_length)
                    if mnist_train[random_index][1] < self.CLASS:
                        random_indices.append(random_index)
                        break
            sampler = torch.utils.data.sampler.SubsetRandomSampler(random_indices)
            return sampler
        if mode == 'test':
            mnist_test = FashionMNIST(root="./data", train=False, transform=trans, download=True)
            total_length = len(mnist_test)
            random_indices = []
            for i in range(self.test_num):
                while True:
                    random_index = np.random.randint(total_length)
                    if mnist_test[random_index][1] < self.CLASS:
                        random_indices.append(random_index)
                        break
            sampler = torch.utils.data.sampler.SubsetRandomSampler(random_indices)
            return sampler

    def preprocessing(self):
        # 使用 DataLoader 创建数据加载器
        sampler_train = self.dataset_load('train')
        sampler_test = self.dataset_load('test')
        train_loader = DataLoader(FashionMNIST(root="./data", train=True, transform=ToTensor(), download=True),
                                  sampler=sampler_train)
        test_loader = DataLoader(FashionMNIST(root="./data", train=True, transform=ToTensor(), download=True),
                                  sampler=sampler_test)

        # 新建数组，存储特征和标签
        train_features = []
        train_label = []
        test_features = []
        test_label = []

        # 获取数据，分离输入数据和标签，并展平
        iterator_train = iter(train_loader)
        iterator_test = iter(test_loader)
        for batch in iterator_train:
            inputs, targets = batch
            # 将train展平
            train_flattened_features = inputs.view(inputs.size(0), -1)
            train_features.append(train_flattened_features)
            train_label.append(targets)
        for batch in iterator_test:
            inputs, targets = batch
            # 将test展平
            test_flattened_features = inputs.view(inputs.size(0), -1)
            test_features.append(test_flattened_features)
            test_label.append(targets)
        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)
        return train_features, train_label, test_features, test_label

    def quantum_encode(self):
        # 编码为量子态，模式在类属性中指定
        # 返回使用的比特数，和
        # 特征和标签都是tensor形式
        train_features, train_label, test_features, test_label = self.preprocessing()
        # 训练集标签和测试集标签写入csv文件
        write_label_to_csv(train_label, self.file_path_train_label)
        write_label_to_csv(test_label, self.file_path_test_label)
        # 选择的编码方式
        # 振幅编码
        if self.encoding_method == 'amplitude_code':
            # num_qubits = self.amplitude_code(train_features, self.file_path_train)
            self.amplitude_code(train_features, self.file_path_train)
        print('编码完成，赞美欧姆弥赛亚')

        # 振幅编码
        # if self.encoding_method == 'graph_convolutional_code':
        #     print('还没编好呢宝')
        #
        # if self.encoding_method == 'ground_state_amplitude_code':
        #     print('还没编好呢宝')

    def amplitude_code(self, features, file_path):
        """
        :param features: 输入的数据集
        :param file_path: 编码存储位置
        :return: 编码使用的qubit数
        """
        # 计算features大小。[0]为行数，[1]为列数
        shape = features.size()
        # 计算振幅编码需要的比特数，输出基本信息
        num_qubits = int(np.ceil(np.log2(shape[1])))
        print('数据样本个数为：', shape[0])
        print('数据特征个数为：', shape[1])
        print('编码所需比特数为：', num_qubits)

        # 大循环，遍历样本对样本特征进行编码
        for f in range(shape[0]):
            # 存储每个样本的编码，每编码一个样本存储一次
            density_matrix_list = []
            # 特征数不足比特数，补0。读取样本后才补0.-
            train_features_padded = np.pad(features, (0, 2 ** num_qubits - len(features[0])), mode='constant')
            # 振幅编码使用了initialize，原理上是直接改变电路的输入为特定数据，相当于先算好，后输入
            # 绘制第一部分的编码电路，每个电路都有一个辅助比特。编码电路直接编码好。
            circuit_encode = QuantumCircuit(num_qubits, num_qubits)
            circuit_encode.initialize(train_features_padded[0], list(range(num_qubits)), True)
            circuit_encode.barrier()
            # circuit_drawer(circuit_encode, output='mpl').show()
            # 绘制第二部分的测量电路
            circuit_measure = QuantumCircuit(num_qubits, num_qubits)
            # 绘制第三部分的合并电路
            circuit_combine = QuantumCircuit(num_qubits, num_qubits)
            circuit_combine.add_register(QuantumRegister(1, 'aux'))

            # 小循环，输出每条电路的结果
            for i in range(num_qubits):
                # 绘制测量电路的阶梯测量
                for j in range(num_qubits):
                    if j != i:
                        circuit_measure.measure([j], 1)
                circuit_measure.barrier()
                # circuit_drawer(circuit_measure, output='mpl').show()
                # 合并编码和测量电路
                circuit_combine.compose(circuit_encode, list(range(num_qubits)), inplace=True)
                circuit_combine.compose(circuit_measure, list(range(num_qubits)), inplace=True)
                # circuit_drawer(circuit_combine, output='mpl').show()
                # 加swap，把提取结果置换到辅助比特上
                circuit_combine.swap(i, num_qubits)
                # circuit_drawer(circuit_combine, output='iqp').show()
                # 运行电路并获取结果
                backend = Aer.get_backend('statevector_simulator')
                result = execute(circuit_combine, backend=backend, shots=1).result()
                # 获取参与量子态向量，迹掉除了辅助比特外的其他比特，得到当前这一条线路的输出
                ancilla_state = result.get_statevector()
                density_matrix_partial = partial_trace(ancilla_state, list(range(0, num_qubits)))
                # 所有一个sample
                density_matrix_list.append(density_matrix_partial)
                # 清空measure和combine电路,开始下一轮绘制
                circuit_measure.clear()
                circuit_combine.clear()
            # 小循环获得十个量子比特编码，相加归一化为一个密度矩阵。输入前检查是否符合密度矩阵条件。
            encode_result = sum_normalize_density_matrices(density_matrix_list)
            # check_matrix_conditions(encode_result)
            write_density_matrices(encode_result, file_path)
