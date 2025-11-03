from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
import numpy as np
from qiskit import BasicAer
from qiskit.quantum_info import DensityMatrix
from constants import (batch_train, Epochs, dim, train_num)


# 创建参数化量子线路的函数
def create_parameterized_circuit(num_qubits):
    # 创建一个量子线路
    qc = QuantumCircuit(num_qubits)

    # 创建参数列表，每个参数代表一个旋转角度
    theta = [Parameter(f'theta_{i}') for i in range(num_qubits)]

    # 在每个量子比特上应用一个参数化的旋转门
    for i in range(num_qubits):
        qc.rx(theta[i], i)
        qc.ry(theta[i], i)
        qc.rz(theta[i], i)

    # 添加一个参数化的纠缠门，比如 CRX 门
    if num_qubits > 1:
        for i in range(num_qubits - 1):
            qc.crx(theta[i], i, i + 1)

    return qc, theta


def create_pure_matrices():
    import math
    num_qubits = int(math.log(dim, 2))
    param_values = {}
    for i in range(num_qubits):
        # 为每个i生成一个0到np.pi之间的随机值
        theta_value = np.random.uniform(0, np.pi)
        param_values[i] = theta_value
    # print(param_values)

    param_circuit, theta = create_parameterized_circuit(num_qubits)

    # 为参数设置具体的值
    param_values = {theta[i]: np.random.uniform(0, np.pi) for i in range(num_qubits)}

    # 绑定参数
    bound_circuit = param_circuit.bind_parameters(param_values)

    backend = Aer.get_backend('statevector_simulator')
    result = execute(bound_circuit, backend=backend, shots=1).result()
    # 获取参与量子态向量，迹掉除了辅助比特外的其他比特，得到当前这一条线路的输出
    ancilla_state = result.get_statevector()
    density_Matrix = DensityMatrix(ancilla_state)
    # density_Matrixs.append(density_Matrix)
    return density_Matrix


