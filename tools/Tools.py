import torch
import numpy as np
import csv
import time
import os
from constants import dim
from qiskit.quantum_info import state_fidelity, DensityMatrix


def create_random_density_matrix(size):
    seed = int((time.time() * 1e9) % 4294967295)  # 获取当前时间的纳秒数，并取模以适应种子范围
    torch.manual_seed(seed)  # 设置随机种子
    # 创建一个随机的 Hermitian 矩阵
    A = torch.randn((size, size), dtype=torch.cfloat)  # 使用复数类型
    A = A + A.T.conj()  # 保证矩阵是 Hermitian
    # 将其归一化为密度矩阵
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.abs(eigvals)
    eigvals /= eigvals.sum()  # 归一化

    # 创建复数类型的对角矩阵
    eigvals_diag = torch.diag(eigvals.to(torch.cfloat))

    # 计算密度矩阵
    density_matrix = torch.matmul(torch.matmul(eigvecs, eigvals_diag), eigvecs.T.conj())
    return density_matrix


def create_random_pure_state_density_matrix(size):
    seed = int((time.time() * 1e9) % 4294967295)  # 获取当前时间的纳秒数，并取模以适应种子范围
    torch.manual_seed(seed)  # 设置随机种子

    # 创建一个随机的复数列向量 |psi>
    psi = torch.randn((size, 1), dtype=torch.cfloat)  # 使用复数类型

    # 对列向量进行归一化处理，使得它是一个单位向量
    psi = psi / torch.norm(psi)

    # 计算密度矩阵 rho = |psi><psi|
    density_matrix = psi @ psi.T.conj()

    return density_matrix


# def matrix_sqrt(mat):
#     """计算 Hermitian 复数矩阵的平方根，并保留梯度"""
#     # 对 Hermitian 矩阵进行特征值分解
#     eigvals, eigvecs = torch.linalg.eigh(mat)
#
#     # 计算特征值的平方根，负特征值处理为 0（数值稳定性处理）
#     sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0))
#
#     # 重构平方根矩阵
#     sqrt_mat = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.conj().T
#     return sqrt_mat


def matrix_sqrt(mat):
    """计算复数矩阵的平方根，并保留梯度"""
    # 使用奇异值分解（SVD）计算矩阵的平方根
    U, S, Vh = torch.linalg.svd(mat)

    # 转换 U, S, Vh 到双精度复数类型
    U = U.to(torch.cdouble)
    S = S.to(torch.cdouble)
    Vh = Vh.to(torch.cdouble)

    # 计算奇异值的平方根
    sqrt_S = torch.diag(torch.sqrt(S))

    # 计算矩阵的平方根
    sqrt_mat = U @ sqrt_S @ Vh
    return sqrt_mat


def fidelity_fun(state1, state2, epsilon=1e-10):
    # 检查输入是否为复数类型
    assert state1.is_complex() and state2.is_complex(), "输入的密度矩阵必须是复数类型"
    rho1 = state1.to(torch.cdouble)
    rho2 = state2.to(torch.cdouble)
    try:
        # 计算密度矩阵的平方根
        sqrt_rho1 = matrix_sqrt(rho1)

        # 计算 sqrt(rho1) * rho2 * sqrt(rho1)
        product = sqrt_rho1 @ rho2 @ sqrt_rho1

        # 计算 product 的平方根
        sqrt_product = matrix_sqrt(product)

        # 计算保真度，取实部的迹
        fidelity = torch.real(torch.trace(sqrt_product))

        # 确保保真度不为零，以避免对数计算出现负无穷
        fidelity = torch.clamp(fidelity, min=epsilon)

        # 计算最终的保真度平方
        fidelity_square = fidelity ** 2
        # print('check source fidelity:', '{:.6f}'.format(fidelity_square))

    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        return torch.tensor(float('inf'), dtype=torch.double, requires_grad=False)

    return fidelity_square


def save_kraus(kraus_list, file_path, dim=dim):
    # Check if the file exists and is not empty
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        # Open the file in write mode to clear its contents
        with open(file_path, 'w', newline='') as csvfile:
            pass  # Just open and close to clear the file

    # Move the header writing logic inside the loop
    wrote_header = False

    for kraus_group in kraus_list:
        for kraus in kraus_group[0]:
            # Check if the header needs to be written
            if not wrote_header:
                with open(file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write headers only if they haven't been written yet
                    if np.iscomplexobj(kraus.data):
                        # If the matrix is complex, write both real and imaginary part headers
                        real_part_header = ['Real Part {}'.format(i) for i in range(dim ** 2)]
                        imag_part_header = ['Imaginary Part {}'.format(i) for i in range(dim ** 2)]
                        writer.writerow(real_part_header + imag_part_header)
                    else:
                        # If the matrix is real, write only real part header
                        real_part_header = ['Real Part {}'.format(i) for i in range(dim ** 2)]
                        writer.writerow(real_part_header)
                    wrote_header = True

            # Extract real and imaginary parts and flatten
            if np.iscomplexobj(kraus.data):
                # If the matrix is complex, save both real and imaginary parts
                flattened_real_part = np.real(kraus.data.flatten())
                flattened_imag_part = np.imag(kraus.data.flatten())
                with open(file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(flattened_real_part.tolist() + flattened_imag_part.tolist())
            else:
                # If the matrix is real, save only the real part
                flattened_real_part = kraus.data.flatten()
                with open(file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(flattened_real_part.tolist())


# def save_kraus(kraus_list, file_path):
#     # Check if the file exists and is not empty
#     if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
#         # Open the file in write mode to clear its contents
#         with open(file_path, 'w', newline='') as csvfile:
#             pass  # Just open and close to clear the file
#
#     # Move the header writing logic inside the loop
#     wrote_header = False
#
#     for kraus_group in kraus_list:
#         for kraus in kraus_group[0]:
#             # Check if the header needs to be written
#             if not wrote_header:
#                 with open(file_path, 'a', newline='') as csvfile:
#                     writer = csv.writer(csvfile)
#                     # Write headers only if they haven't been written yet
#                     n = dim  # Make sure 'dim' is defined in this scope
#                     real_part_header = ['Real Part {}'.format(i) for i in range(n ** 2)]
#                     imag_part_header = ['Imaginary Part {}'.format(i) for i in range(n ** 2)]
#                     writer.writerow(real_part_header + imag_part_header)
#                     wrote_header = True
#
#             # Extract real and imaginary parts and flatten
#             flattened_real_part = np.real(kraus.data.flatten())
#             flattened_imag_part = np.imag(kraus.data.flatten())
#             # Append real and imaginary parts
#             with open(file_path, 'a', newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(flattened_real_part.tolist() + flattened_imag_part.tolist())


def load_kraus(file_path):
    density_matrix_list = []
    # 读取CSV文件
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # 跳过表头
        headers = next(reader)
        # 密度矩阵的维度
        n = int(len(headers) / 2)
        d = dim
        # d = int(len(headers) / dim)
        for row in reader:
            # 从每一行中提取展平后的实部和虚部
            flattened_real_part = np.array(row[:n], dtype=float)
            flattened_imag_part = np.array(row[n:], dtype=float)
            try:
                # 将展平后的实部和虚部重建为复数矩阵
                complex_matrix = flattened_real_part + 1j * flattened_imag_part
                complex_matrix = complex_matrix.reshape((d, d))

                # 构造密度矩阵并添加到列表
                density_matrix_list.append(complex_matrix)
            except ValueError as e:
                print(f"Error reading row: {row}. {e}")

    return density_matrix_list


def write_density_matrices(density_matrix, file_path):
    # 如果文件不存在，则创建文件并写入表头
    if not os.path.isfile(file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头，实数和虚数部分各占一行
            n = dim
            real_part_header = ['Real Part {}'.format(i) for i in range(n ** 2)]
            imag_part_header = ['Imaginary Part {}'.format(i) for i in range(n ** 2)]
            writer.writerow(real_part_header + imag_part_header)
    # 提取实部和虚部并展平
    flattened_real_part = np.real(density_matrix.data.flatten())
    flattened_imag_part = np.imag(density_matrix.data.flatten())
    # 追加写入展平后的实部和虚部
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(flattened_real_part.tolist() + flattened_imag_part.tolist())
        # print(f'写入成功~{file_path}')


# 从csv中读出矩阵并重建。返回读取函数值
def read_density_matrices(file_path):
    density_matrix_list = []
    # 读取CSV文件
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # 跳过表头
        headers = next(reader)
        # 密度矩阵的维度
        n = int(len(headers) / 2)
        d = dim
        # d = int(len(headers) / dim)
        for row in reader:
            # 从每一行中提取展平后的实部和虚部
            flattened_real_part = np.array(row[:n], dtype=float)
            flattened_imag_part = np.array(row[n:], dtype=float)
            try:
                # 将展平后的实部和虚部重建为复数矩阵
                complex_matrix = flattened_real_part + 1j * flattened_imag_part
                complex_matrix = complex_matrix.reshape((d, d))

                # 构造密度矩阵并添加到列表
                density_matrix = DensityMatrix(complex_matrix)
                density_matrix_list.append(density_matrix)
            except ValueError as e:
                print(f"Error reading row: {row}. {e}")

    return density_matrix_list


# 把标签写入csv。也可用于写入任何张量
def write_label_to_csv(input, file_path):
    # 将 PyTorch Tensor 转换为 NumPy 数组，然后再转换为 Python 列表
    if isinstance(input, torch.Tensor):  # 如果 input 是 PyTorch 张量
        numpy_data = input.numpy().tolist()
    elif isinstance(input, np.ndarray):  # 如果 input 是 NumPy 数组
        numpy_data = input.tolist()
    else:  # 如果 input 既不是张量也不是数组
        numpy_data = input

    # numpy_data = input.numpy().tolist()
    # 如果输出文件夹不存在，则创建
    output_folder = os.path.dirname(file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 将数据写入 CSV 文件
    with open(file_path, 'w', newline='') as csvfile:
        # 逐行写入数据
        writer = csv.writer(csvfile)
        for item in numpy_data:
            writer.writerow([item])


# 从csv中读取标签。返回重建后的标签张量
def read_label(file_path):
    # 读取CSV文件并将数据还原为 NumPy 数组
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    # 将数据转换为 NumPy 数组和。
    numpy_data = np.array([[float(entry) for entry in row] for row in data])

    # 将 NumPy 数组转换为 PyTorch Tensor
    tensor_data = torch.tensor(numpy_data)

    return tensor_data


# 多个矩阵相加后归一化。相加前矩阵保迹，相加后也保迹且厄米
def sum_normalize_density_matrices(density_matrices):
    # 将密度矩阵相加
    total_matrix = sum(dm.data for dm in density_matrices)
    # 计算迹
    trace = np.trace(total_matrix)
    # 归一化并确保是密度矩阵
    normalized_matrix = total_matrix / trace
    normalized_density_matrix = DensityMatrix(normalized_matrix)
    return normalized_density_matrix


def check_density_matrix(rho):
    """
    检查量子态密度矩阵是否满足条件
    :param rho: 输入的量子态密度矩阵
    :return: True表示满足条件，False表示不满足条件
    """
    # 检查是否是方阵
    if rho.shape[0] != rho.shape[1]:
        print('不是方阵')
        return False

    # 检查是否是厄米特矩阵
    if not torch.allclose(rho, rho.conj().T):
        print("不满足厄密矩阵")
        return False

    # 检查是否是正定矩阵
    eigvals, _ = torch.linalg.eigh(rho, UPLO='U')  # 使用UPLO='U'指定使用上三角部分
    if not torch.all(eigvals >= 0):
        print("不满足非负定")
        return False

    if not abs(torch.trace(rho).item() - 1.0) <= 1e-3:
        print("不满足迹为一", torch.trace(rho).item())
        return False
    return True


# 用于检测任一密度矩阵时候符合密度矩阵条件。注意只能输入DensityMatrix
def check_matrix_conditions(matrix):
    eps = 1e-5
    matrix = matrix.data
    # 检查迹是否接近于1
    if np.trace(matrix) >= 1 + eps or np.trace(matrix) <= 1 - eps:
        print('matrix', matrix)
        raise ValueError("初始化失败，矩阵不满足迹为1的条件！")
    # 检查矩阵是否是厄米矩阵
    if not np.allclose(np.conj(matrix.T), matrix):
        print('matrix', matrix)
        raise ValueError("初始化失败，不满足厄米矩阵的条件！")
    print("输入的矩阵符合条件！")


def check_kraus_condition(kraus_operators, tol=1e-6):
    n = kraus_operators[0].size(0)
    identity_matrix = torch.eye(n, dtype=torch.cdouble)
    sum_kraus = torch.zeros((n, n), dtype=torch.cdouble)

    # 计算 \sum_{m} K_{m}^{+} K_{m}
    for kraus_op in kraus_operators:
        sum_kraus += torch.matmul(kraus_op.conj().t(), kraus_op)

    # 检查 \sum_{m} K_{m}^{+} K_{m} 是否等于单位矩阵
    if torch.allclose(sum_kraus, identity_matrix, atol=tol):
        print("Kraus operators satisfy the condition.")
        return True
    else:
        print("Kraus operators do NOT satisfy the condition.")
        return False


#  矩阵算共轭转置
def dagger(M):
    return np.transpose(np.conj(M))


def Linear_Dependent(A):
    """
    判断向量是否线性相关
    :param A: 待判断的向量
    :return: True表示线性相关，False表示线性无关
    """
    Q, R = np.linalg.qr(A)
    rank = np.linalg.matrix_rank(R)
    if rank == np.min(A.shape):
        return False  # 无线性相关
    else:
        return True  # 有线性相关
