import numpy as np
import torch
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

import Generate_diffusion_model
from constants import (filepath, pre_layers, tau_perlayer, tau_train_all,
                       Kraus_PERLAYES, Epochs, dim, train_num, lambda_l, pT, Epochs_constrained)
from models import HQMM_model
from tools.Tools import (matrix_sqrt, save_kraus, write_density_matrices,
                         load_kraus, read_density_matrices, create_random_pure_state_density_matrix)
from tools.pure_state import create_pure_matrices


def fidelity_loss(state1, state2, epsilon=1e-10):
    loss = 0
    state_tensors = []
    fidelitys = 0

    for state in state2:
        state_tensor1 = torch.tensor(state, dtype=torch.cdouble, requires_grad=False)
        state_tensors.append(state_tensor1)

    for rho1, rho2 in zip(state1, state_tensors):
        assert rho1.is_complex() and rho2.is_complex(), "输入的密度矩阵必须是复数类型"  # 检查输入是否为复数类型
        # assert torch.allclose(rho1, rho1.conj().T), f"rho1 is not Hermitian{torch.trace(rho1)}"
        # assert torch.allclose(rho2, rho2.conj().T), f"rho2 is not Hermitian{torch.trace(rho2)}"
        # assert torch.isclose(torch.trace(rho1), torch.tensor(1.0, dtype=rho1.dtype)), "Trace of rho1 is not 1"
        # assert torch.isclose(torch.trace(rho2), torch.tensor(1.0, dtype=rho2.dtype)), "Trace of rho2 is not 1"

        rho1 = rho1.to(torch.cdouble)
        rho2 = rho2.to(torch.cdouble)
        try:
            sqrt_rho1 = matrix_sqrt(rho1)  # 计算密度矩阵的平方根
            product = sqrt_rho1 @ rho2 @ sqrt_rho1  # 计算 sqrt(rho1) * rho2 * sqrt(rho1)
            sqrt_product = matrix_sqrt(product)  # 计算 product 的平方根
            fidelity = torch.real(torch.trace(sqrt_product))  # 计算保真度，取实部的迹
            fidelity = (torch.clamp(fidelity, min=epsilon)) ** 2  # 确保保真度不为零，以避免对数计算出现负无穷
            fidelitys += fidelity
            loss += -torch.log(fidelity)  # 计算损失函数
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            return torch.tensor(float('inf'), dtype=torch.double, requires_grad=False)
    return loss / len(state2), fidelitys / len(state2)


def create_train_data():
    for _ in range(train_num):
        density_matrix = create_pure_matrices()
        write_density_matrices(density_matrix, f'{filepath}/source_desity_matrix.csv')


def load_data():
    create_train_data()
    density_matrices = read_density_matrices(fr'{filepath}/source_desity_matrix.csv')
    x = density_matrices

    x_np = np.array([np.array(a) for a in x])
    n = x_np.shape[0]

    # 计算按照8:2比例分割的索引
    split_index = int(n * 0.8)

    # 生成一个随机排列的索引数组
    indices = np.random.permutation(n)

    # 分割数组
    group1_indices = indices[:split_index]
    group2_indices = indices[split_index:]

    # 根据随机索引分配数组
    x_train = x_np[group1_indices]
    x_test = x_np[group2_indices]

    x_train_list = x_train.tolist()
    x_test_list = x_test.tolist()

    return x_train_list, x_test_list


def Plot(fidelitys_Plot):
    x = list(range(len(fidelitys_Plot)))
    fidelitys_Plot_transposed = list(map(list, zip(*fidelitys_Plot)))
    plt.figure()
    for i, fidelity_data in enumerate(fidelitys_Plot_transposed):
        plt.plot(x, fidelity_data, label=f'Line {i + 1}')
    plt.legend()
    plt.title('Fidelity Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Fidelity')
    plt.show()

    with open(fr'{filepath}/fidelitys_all_constrained.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(fidelitys_Plot_transposed)


def plot(losses, fidelitys, l):
    x = list(range(len(losses)))

    # 创建一个图形和两个子图对象
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1行2列

    # 在 ax1 上绘制损失值 (losses)
    ax1.plot(x, losses, label='Losses', color='b')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Loss Over Iterations')
    ax1.legend(loc='upper left')

    # 在 ax2 上绘制保真度值 (fidelities)
    ax2.plot(x, fidelitys, label='Fidelities', color='r')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fidelity', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Fidelity Over Iterations')
    ax2.axhline(y=1, color='r', linestyle='--', linewidth=1, label='y=1 (Fidelity)')
    ax2.legend(loc='upper left')

    # 找到最大保真度和最小损失的索引，并打印
    max_fid_index = fidelitys.index(max(fidelitys))
    min_loss_index = losses.index(min(losses))
    print(f'Max Fidelity: {max(fidelitys):.6g}')
    print(f'Min Loss: {min(losses):.6g}')

    # 标注最大保真度和最小损失
    ax1.text(min_loss_index, min(losses), f'Min Loss: {min(losses):.6g}', fontsize=10,
             verticalalignment='top', color='b')
    ax2.text(max_fid_index, max(fidelitys), f'Max Fidelity: {max(fidelitys):.6g}', fontsize=10,
             verticalalignment='bottom', color='r')

    fig.tight_layout()

    # 保存图像和CSV文件
    plt.savefig(fr'{filepath}/plt_{l}.png')
    # plt.show()

    data = {'losses': losses, 'fidelities': fidelitys}
    df = pd.DataFrame(data)
    df.index = range(1, len(df) + 1)
    df.to_csv(fr'{filepath}/fidelitys_{l}.csv', index=True)


from scipy.linalg import sqrtm


def trace_distance(rho, sigma):
    """
    Calculate the trace distance between two quantum states rho and sigma.

    Parameters:
    rho (numpy array): The density matrix of the first quantum state.
    sigma (numpy array): The density matrix of the second quantum state.

    Returns:
    float: The trace distance between rho and sigma.
    """
    # Ensure the input matrices are square and Hermitian
    assert rho.shape == sigma.shape, "Matrices must be square and of the same size."
    assert np.allclose(rho, rho.conj().T) and np.allclose(sigma, sigma.conj().T), "Matrices must be Hermitian."

    # Calculate the difference between the two density matrices
    delta = rho - sigma

    # # Check if the matrix is positive semi-definite
    # if not np.all(np.linalg.eigvals(delta_product) >= 0):
    #     print("rho, sigma", rho, sigma)
    #     raise ValueError("The matrix delta^dagger * delta is not positive semi-definite.")

    # Calculate the square root of the matrix delta^dagger * delta
    sqrt_delta = sqrtm(delta.conj().T @ delta)

    # Calculate the trace of the square root matrix
    trace_sqrt_delta = np.trace(sqrt_delta)

    # The trace distance is half the trace of the square root matrix
    return 0.5 * trace_sqrt_delta


def train(QHMNN, x_train_list):
    # # 倒序逐层解除噪声
    # for l in range(pre_layers, pre_layers - 1, -1):
    #     x_kraus_L = QHMNN.forward_pre(x_train_list, l)
    #     x_kraus_l = QHMNN.forward_pre(x_train_list, l - 1)
    #     X_kraus_L = np.array([np.array(a) for a in x_kraus_L])
    #     X_kraus_l = np.array([np.array(a) for a in x_kraus_l])
    #     X_kraus_L_tensor = torch.tensor(X_kraus_L, dtype=torch.complex64, requires_grad=True)
    #     X_kraus_l_tensor = torch.tensor(X_kraus_l, dtype=torch.complex64, requires_grad=True)
    #
    #     # Calculate the trace distance
    #     distance = 0
    #     for i in range(len(X_kraus_L)):
    #         distance += trace_distance(X_kraus_L[i], X_kraus_l[i])
    #     print(f"The trace distance between {l} and {l - 1} is:", distance / len(X_kraus_L))
    #
    #     losses = []
    #     fidelitys = []
    #     for epoch in range(Epochs):
    #         time1 = time.perf_counter()
    #         output_kraus_tensor = QHMNN.forward(X_kraus_L_tensor)  # 前向传播
    #         loss, fidelity = fidelity_loss(output_kraus_tensor, X_kraus_l_tensor)
    #         gradients = torch.autograd.grad(loss, list(QHMNN.parameters()), retain_graph=True)
    #         with torch.no_grad():
    #             QHMNN.update_kraus_operators(gradients)
    #         losses.append(loss.item())
    #         fidelitys.append(fidelity.item())
    #         print(f"Epoch {epoch}: Loss = {loss.item()}, fidelity = {fidelity.item()}, "
    #               f"per_epoch_time: {time.perf_counter() - time1}")
    #
    #     plot(losses, fidelitys, l)
    #     kraus_operators = []
    #     for idx, param in enumerate(QHMNN.kraus_operators):
    #         kraus_operator = []
    #         for p in param:
    #             kraus_operator.append(p.data)
    #         kraus_operators.append((torch.stack(kraus_operator)).view(1, Kraus_PERLAYES, dim, dim))
    #     save_kraus(kraus_operators, f'{filepath}/kraus_operators_array_{l}.csv')
    #
    # # # ======================train all layers constrained====================
    QHMM_All_constrained = HQMM_model.QHMM_train_all_constrained()
    X_kraus_l = np.array([np.array(a) for a in x_train_list])
    X_kraus_l_tensor = torch.tensor(X_kraus_l, dtype=torch.complex64, requires_grad=True)

    noise_kraus = []
    for l in range(pre_layers):  # 噪声从0-pre_layers
        noise_kraus.append(
            torch.tensor(np.array([np.array(a) for a in QHMNN.forward_pre(x_train_list, l)]), dtype=torch.complex64,
                         requires_grad=True))

    noise_kraus_less = [X_kraus_l_tensor]
    for l in range(pre_layers - 1):  # 噪声从0-pre_layers-1
        noise_kraus_less.append(
            torch.tensor(np.array([np.array(a) for a in QHMNN.forward_pre(x_train_list, l)]), dtype=torch.complex64,
                         requires_grad=False))

    losses = []
    fidelitys = []
    fidelitys_Plot = []
    for epoch in range(Epochs_constrained):
        time1 = time.perf_counter()
        output_kraus_tensors = QHMM_All_constrained.forward_all(noise_kraus)  # 前向传播
        loss_all = torch.tensor(0)
        fidelity_all = 0
        fidelitys_plot = []
        for idx, (output_kraus_tensor, X_kraus_l) in enumerate(zip(output_kraus_tensors, noise_kraus_less)):
            loss, fidelity = fidelity_loss(output_kraus_tensor, X_kraus_l)
            print("idx fidelity:", f'{idx}： {fidelity}')
            loss_all = loss_all + loss * lambda_l[idx]
            # loss_all = loss_all + loss
            fidelity_all += fidelity.item()
            fidelitys_plot.append(fidelity.item())
        fidelitys_Plot.append(fidelitys_plot)

        gradients = torch.autograd.grad(loss_all, list(QHMM_All_constrained.parameters()), retain_graph=True)
        with torch.no_grad():
            QHMM_All_constrained.update_kraus_operators_all(gradients)
        losses.append(loss_all.item())
        fidelitys.append(fidelity_all / pre_layers)
        print(f"train_all_constrained: epoch {epoch}: Loss = {loss_all.item()}, fidelity = {fidelity_all / pre_layers}, "
              f"per_epoch_time: {time.perf_counter() - time1}")
        QHMM_All_constrained.all_state_tensors = None
        QHMM_All_constrained.zero_grad()

    Plot(fidelitys_Plot)

    plot(losses, fidelitys, -1)
    kraus_operators = []
    for idx, param in enumerate(QHMM_All_constrained.kraus_operators_all):
        kraus_operator_all = []
        for p in param:
            kraus_operator_all.append(p.data)
        kraus_operators.append((torch.stack(kraus_operator_all)).view(1, Kraus_PERLAYES, dim, dim))

    save_kraus(kraus_operators, f'{filepath}/kraus_operators_array_0_constrained.csv')


def main():
    QHMNN = HQMM_model.quantum_hidden_Markov_neural_network()
    x_train_list, x_test_list = load_data()
    train(QHMNN, x_train_list)

    # test_model
    X_kraus_l = np.array([np.array(a) for a in x_train_list])
    rho_0 = torch.tensor(X_kraus_l, dtype=torch.complex64, requires_grad=False)

    x_kraus_L = QHMNN.forward_pre(x_train_list, pre_layers)  # 最后一层的加噪数据
    G_diffusion_model = Generate_diffusion_model.Generating_diffusion_model()
    filename = fr'{filepath}/final_check.txt'
    with torch.no_grad():
        output_kraus_tensor1 = G_diffusion_model.generation_forward(x_kraus_L)  # 完全解躁
        loss, fidelity = fidelity_loss(output_kraus_tensor1, rho_0)
        print(f"第一遍：Loss = {loss.item()}, fidelity = {fidelity.item()}")
        data = {
            "loss": loss.item(),
            "fidelity": fidelity.item(),
            "epoch": Epochs,
            "tau_perlayer": tau_perlayer,
            "tau_train_all": tau_train_all,
            "dim": dim,
            "pt": pT,
        }
        with open(filename, "w") as file:
            for key, label in data.items():
                file.write(f"{label}: {key}\n")

    # ===========================带约束的测试===========================
    G_diffusion_model_all_constrained = Generate_diffusion_model.Generating_diffusion_model_all_constrained()
    # 加载数据集
    X_kraus_l = np.array([np.array(a) for a in x_train_list])
    X_kraus_l_tensor = torch.tensor(X_kraus_l, dtype=torch.complex64, requires_grad=False)
    noise_kraus_less = [X_kraus_l_tensor]
    for l in range(pre_layers - 1):  # 噪声从0-pre_layers-1
        noise_kraus_less.append(
            torch.tensor(np.array([np.array(a) for a in QHMNN.forward_pre(x_train_list, l)]), dtype=torch.complex64,
                         requires_grad=False))

    noise_kraus = []
    for l in range(pre_layers):  # 噪声从0-pre_layers
        noise_kraus.append(
            torch.tensor(np.array([np.array(a) for a in QHMNN.forward_pre(x_train_list, l)]), dtype=torch.complex64,
                         requires_grad=False))

    loss_all = 0
    fidelity_all = 0
    fidelity_group = []
    loss_all_group = []
    filename = fr'{filepath}/final_check_all_constrained.txt'
    with torch.no_grad():
        rho_last = G_diffusion_model_all_constrained.generation_forward(noise_kraus)
        for idx, (output_kraus_tensor, X_kraus_l) in enumerate(zip(rho_last, noise_kraus_less)):
            loss, fidelity = fidelity_loss(output_kraus_tensor, X_kraus_l)
            loss_all = loss_all + loss * lambda_l[idx]
            fidelity_all += fidelity.item()
            fidelity_group.append(fidelity.item())
            loss_all_group.append(loss.item())

        print(f"整体训练结果：Loss = {loss_all_group}, fidelity = {fidelity_group}")
        data = {
            "Overall training results：loss": loss_all_group,
            "fidelity": fidelity_group,
            "epoch": Epochs,
            "tau_perlayer": tau_perlayer,
            "tau_train_all": tau_train_all,
            "dim": dim,
            "pt": pT,
            "lambda": lambda_l,
        }
        with open(filename, "w") as file:
            for key, label in data.items():
                file.write(f"{label}: {key}\n")


if __name__ == "__main__":
    main()
