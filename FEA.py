import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
# import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
import cvxopt
import cvxopt.cholmod


# 有限元分析部分
def finite_element_analysis(Real_xPhys_FEA, F, bc):
    nelx = 128
    nely = 64
    penal = 3
    E0 = 1.0
    Emin = 1e-9
    nu = 0.3
    ndof = len(bc)
    fixed = np.where(bc == 1)[0]  # np.where 返回满足条件的索引
    free = np.setdiff1d(np.arange(ndof), fixed)
    k = np.array(
        [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
         nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E0 / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])  # 刚度矩阵

    edofMat = np.zeros((nelx * nely, 8), dtype=int)  # 初始化自由度矩阵，有nelx*nely个单元，每个单元8个自由度（2*4）
    for elx in range(nelx):  # 自由度编码
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])

    iK = np.kron(edofMat, np.ones((8, 1))).flatten()  # 组装总刚需要
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # 定义有限元求解
    u = np.zeros((ndof, 1))  # 初始化位移向量
    xPhys_flattened = np.reshape(Real_xPhys_FEA, [nelx * nely], order='F') # order='F' 参数（表示 Fortran-like order，即列优先顺序）

    # solve求解
    sK = ((KE.flatten()[np.newaxis]).T *  (Emin + xPhys_flattened ** penal * (E0 - Emin))).flatten(order='F')
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc() # 构造稀疏刚度矩阵（COO 格式


    def deleterowcol(A, delrow, delcol):
        # Assumes that matrix is in symmetric csc form !
        m = A.shape[0]
        keep = np.delete(np.arange(0, m), delrow)
        A = A[keep, :]
        keep = np.delete(np.arange(0, m), delcol)
        A = A[:, keep]
        return A

    K = deleterowcol(K, fixed, fixed).tocoo() #  删除固定自由度
    K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int)) # 将稀疏矩阵转换为 cvxopt 格式，用于求解
    K += cvxopt.spmatrix(1e-8, range(K.size[0]), range(K.size[1]))
    B = cvxopt.matrix(F[free, 0].flatten())

    # B = cvxopt.matrix(F[free, 0]) # 将载荷向量 self.f 转换为 cvxopt 矩阵。
    cvxopt.cholmod.linsolve(K, B) # 使用稀疏 Cholesky 分解快速求解线性方程组。
    u[free, 0] = np.array(B)[:, 0]
    u = np.abs(u)

    # Von Mises 应力计算
    q = 1
    L = 0.01
    BS = (1 / (2 * L)) * torch.tensor([[-1, 0, 1, 0, 1, 0, -1, 0],
                                       [0, -1, 0, -1, 0, 1, 0, 1],
                                       [-1, -1, -1, 1, 1, 1, 1, -1]])
    D = (1 / (1 - nu**2)) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    # 计算单元应力张量 S
    tolne = nelx*nely  # 单元总数
    S = np.zeros((tolne, 3))  # 初始化应力张量
    for i in range(tolne):
        u_e = u[edofMat[i, :], 0]  # 提取单元自由度的位移
        S[i, :] = (D @ BS @ u_e) * (xPhys_flattened[i] ** q)  # 应力张量修正

    # 计算 von Mises 应力
    S1, S2, S12 = S[:, 0], S[:, 1], S[:, 2]
    # von_mises = np.sqrt(S1 ** 2 - S1 * S2 + S2 ** 2 + 3 * S12 ** 2)
    von_mises = np.sqrt(np.clip(S1 ** 2 - S1 * S2 + S2 ** 2 + 3 * S12 ** 2, 1e-6, None))

    # 转换 von_mises 为 PyTorch Tensor
    von_mises_tensor = torch.tensor(von_mises, dtype=torch.float32)
    # von_mises_tensor = von_mises_tensor.reshape(64, 128)
    von_mises_tensor = von_mises_tensor.reshape(128, 64).T

    # von_mises_numpy = von_mises_tensor.cpu().numpy()  # 确保它在 CPU 上
    #
    # # 绘制 von Mises 应力分布
    # plt.figure(figsize=(10, 5))
    # plt.imshow(von_mises_numpy, cmap='jet', aspect='equal')
    # plt.colorbar(label="Von Mises Stress")  # 添加颜色条
    # plt.title("Von Mises Stress Distribution (Tensor)")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()

    # 计算目标函数
    p = 8  # Example p-norm parameter
    obj = torch.sum(von_mises_tensor ** p) ** (1 / p)

    return von_mises_tensor, obj

# 体积分数绝对误差函数
def calculate_volume_fraction_loss(fake_xPhys, real_xPhys): #  fake_xPhys 的形状是 (2, 1, 64, 128)
    """计算生成器的体积分数绝对误差损失 (AE^{VF}_G)。"""
    batch_size, _, height, width = fake_xPhys.shape # (2,1,64,128)
    n = height * width  # 总像素数
    fake_vf = torch.sum(fake_xPhys, dim=(2, 3)) / n  # 假样本总体积分数  fake_vf 的形状是 (batch_size, 1)
    real_vf = torch.sum(real_xPhys, dim=(2, 3)) / n  # 真实样总本体积分数 形状是 (batch_size, 1)
    return torch.abs(fake_vf - real_vf).mean(), fake_vf, real_vf


# finite_element_analysis 函数，使其可以直接处理批量数据
def finite_element_analysis_batch(Real_xPhys_FEA_batch, F_batch, bc_batch):
    batch_size = Real_xPhys_FEA_batch.shape[0]
    von_mises_tensor = []
    objs = []

    for i in range(batch_size):
        Real_xPhys_FEA = Real_xPhys_FEA_batch[i]
        F = F_batch[i]
        bc = bc_batch[i]

        von_mises_tensor, obj = finite_element_analysis(Real_xPhys_FEA, F, bc)

        von_mises_tensor.append(von_mises_tensor)
        objs.append(obj)

    return von_mises_tensor, torch.tensor(objs, dtype=torch.float32)


# 评价指标计算函数
def evaluate_metrics(fake_xPhys, real_xPhys, fake_vf, real_vf, Real_xPhys_FEA, Fake_xPhys_FEA, bc, F, batch_size):
    """计算评价指标：MAE、MSE、AE^{VF}_G、RE^{VF}_G 和 von MISES 误差"""
    # MAE 和 MSE,逐点误差,反映生成器的预测密度分布与真实分布的逐点拟合效果。
    fake_xPhys_flat = fake_xPhys.view(batch_size, -1)  # 展平
    real_xPhys_flat = real_xPhys.view(batch_size, -1)
    mae = torch.mean(torch.abs(fake_xPhys_flat - real_xPhys_flat))
    mse = torch.mean((fake_xPhys_flat - real_xPhys_flat) ** 2)

    # 像素精度 Pixel-error 空间匹配度评价指标
    fake_binary = (fake_xPhys >= 0.5).float()
    real_binary = (real_xPhys >= 0.5).float()
    correct_pixels = (fake_binary == real_binary).sum().item()
    total_pixels = real_binary.numel()
    accuracy = correct_pixels / total_pixels

    # # vf相对误差,全局体积分数误差,验证生成器是否在满足全局设计约束的情况下生成结构。
    re_vf = torch.mean(torch.clamp(torch.abs(fake_vf - real_vf) / real_vf))

    # # 决定系数，越接近1相关性越强
    r2_vf = 1 - torch.sum((fake_vf - real_vf) ** 2) / torch.sum((real_vf - torch.mean(fake_vf + real_vf)) ** 2 + 1e-8)

    #  von Mises 应力相对误差
    # 此处需要计算Fake_xPhys和Real_xPhys在相对应的载荷下全局应力obj相对误差。
    Real_vm,_ = finite_element_analysis_batch(Real_xPhys_FEA, F, bc)  #Tensor（8192，）
    Fake_vm,_ = finite_element_analysis_batch(Fake_xPhys_FEA, F, bc)


    # 计算目标函数
    p = 8  # Example p-norm parameter
    Real_obj = torch.sum(Real_vm ** p) ** (1 / p)
    Fake_obj = torch.sum(Fake_vm ** p) ** (1 / p)
    re_obj = torch.mean(torch.clamp(torch.abs(Fake_obj - Real_obj) / (Real_obj + 1e-6)))

    return mae.item(), mse.item(),  re_vf.item(), re_obj.item(), accuracy, r2_vf,Real_vm , Fake_vm

# 计算应力分布及p范数应力
def stress_distribution(Real_xPhys_FEA, F, bc):
    nelx = 128
    nely = 64
    penal = 3
    E0 = 1.0
    Emin = 1e-9
    nu = 0.3
    ndof = len(bc)
    fixed = np.where(bc == 1)[0]  # np.where 返回满足条件的索引
    free = np.setdiff1d(np.arange(ndof), fixed)
    k = np.array(
        [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
         nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E0 / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])  # 刚度矩阵

    edofMat = np.zeros((nelx * nely, 8), dtype=int)  # 初始化自由度矩阵，有nelx*nely个单元，每个单元8个自由度（2*4）
    for elx in range(nelx):  # 自由度编码
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])

    iK = np.kron(edofMat, np.ones((8, 1))).flatten()  # 组装总刚需要
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # 定义有限元求解
    u = np.zeros((ndof, 1))  # 初始化位移向量
    xPhys_flattened = np.reshape(Real_xPhys_FEA, [(nelx) * nely], order='F') # order='F' 参数（表示 Fortran-like order，即列优先顺序）

    # solve求解
    sK = ((KE.flatten()[np.newaxis]).T *  (Emin + xPhys_flattened ** penal * (E0 - Emin))).flatten(order='F')
    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc() # 构造稀疏刚度矩阵（COO 格式


    def deleterowcol(A, delrow, delcol):
        # Assumes that matrix is in symmetric csc form !
        m = A.shape[0]
        keep = np.delete(np.arange(0, m), delrow)
        A = A[keep, :]
        keep = np.delete(np.arange(0, m), delcol)
        A = A[:, keep]
        return A

    K = deleterowcol(K, fixed, fixed).tocoo() #  删除固定自由度
    K = cvxopt.spmatrix(K.data, K.row.astype(int), K.col.astype(int)) # 将稀疏矩阵转换为 cvxopt 格式，用于求解
    K += cvxopt.spmatrix(1e-8, range(K.size[0]), range(K.size[1]))
    B = cvxopt.matrix(F[free, 0].flatten())

    # B = cvxopt.matrix(F[free, 0]) # 将载荷向量 self.f 转换为 cvxopt 矩阵。
    cvxopt.cholmod.linsolve(K, B) # 使用稀疏 Cholesky 分解快速求解线性方程组。
    u[free, 0] = np.array(B)[:, 0]
    u = np.abs(u)

    # Von Mises 应力计算
    q = 1
    L = 0.01
    BS = (1 / (2 * L)) * torch.tensor([[-1, 0, 1, 0, 1, 0, -1, 0],
                                       [0, -1, 0, -1, 0, 1, 0, 1],
                                       [-1, -1, -1, 1, 1, 1, 1, -1]])
    D = (1 / (1 - nu**2)) * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])

    # 计算单元应力张量 S
    tolne = nelx*nely  # 单元总数
    S = np.zeros((tolne, 3))  # 初始化应力张量
    for i in range(tolne):
        u_e = u[edofMat[i, :], 0]  # 提取单元自由度的位移
        S[i, :] = (D @ BS @ u_e) * (xPhys_flattened[i] ** q)  # 应力张量修正

    # 计算 von Mises 应力
    S1, S2, S12 = S[:, 0], S[:, 1], S[:, 2]
    # von_mises = np.sqrt(S1 ** 2 - S1 * S2 + S2 ** 2 + 3 * S12 ** 2)
    von_mises = np.sqrt(np.clip(S1 ** 2 - S1 * S2 + S2 ** 2 + 3 * S12 ** 2, 1e-6, None))


    # 转换 von_mises 为 PyTorch Tensor

    von_mises_tensor = torch.tensor(von_mises, dtype=torch.float32)
    von_mises_tensor = von_mises_tensor.reshape(128, 64).T

    # von_mises_numpy = von_mises_tensor.cpu().numpy()  # 确保它在 CPU 上
    #
    # # 绘制 von Mises 应力分布
    # plt.figure(figsize=(10, 5))
    # plt.imshow(von_mises_numpy, cmap='jet', aspect='auto')
    # plt.colorbar(label="Von Mises Stress")  # 添加颜色条
    # plt.title("Von Mises Stress Distribution (Tensor)")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()

    # 计算目标函数
    p = 8  # Example p-norm parameter
    obj = torch.sum(von_mises_tensor ** p) ** (1 / p)

    return obj ,von_mises_tensor


# 评价指标计算函数
def evaluate_metrics_train(fake_xPhys, real_xPhys, fake_vf, real_vf, Real_xPhys_FEA, Fake_xPhys_FEA, bc, F, batch_size):
    """计算评价指标：MAE、MSE、AE^{VF}_G、RE^{VF}_G 和 von MISES 误差"""
    # MAE 和 MSE,逐点误差,反映生成器的预测密度分布与真实分布的逐点拟合效果。
    fake_xPhys_flat = fake_xPhys.view(batch_size, -1)  # 展平
    real_xPhys_flat = real_xPhys.view(batch_size, -1)
    mae = torch.mean(torch.abs(fake_xPhys_flat - real_xPhys_flat))
    mse = torch.mean((fake_xPhys_flat - real_xPhys_flat) ** 2)

    # 像素精度 Pixel-error 空间匹配度评价指标
    fake_binary = (fake_xPhys >= 0.5).float()
    real_binary = (real_xPhys >= 0.5).float()
    correct_pixels = (fake_binary == real_binary).sum().item()
    total_pixels = real_binary.numel()
    accuracy = correct_pixels / total_pixels

    # # vf相对误差,全局体积分数误差,验证生成器是否在满足全局设计约束的情况下生成结构。
    re_vf = torch.mean(torch.clamp(torch.abs(fake_vf - real_vf) / real_vf))

    # # 决定系数，越接近1相关性越强
    r2_vf = 1 - torch.sum((fake_vf - real_vf) ** 2) / torch.sum((real_vf - torch.mean(fake_vf + real_vf)) ** 2 + 1e-8)

    return mae.item(), mse.item(),  re_vf.item(), r2_vf.item(), accuracy