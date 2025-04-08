from scipy.stats import vonmises
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import numpy as np
import torch.optim as optim
from FEA import evaluate_metrics
from FEA import evaluate_metrics_train
from FEA import calculate_volume_fraction_loss
from FEA import stress_distribution
import time
import random
import torch.optim.lr_scheduler as lr_scheduler

class TopologyDataset(Dataset):
    def __init__(self, file_list, nelx=128, nely=64):
        self.data = []
        for file in file_list:
            mat_data = sio.loadmat(file)
            Real_xPhys = mat_data['Real_xPhys'].reshape((nelx, nely)).T
            vf = mat_data['vf']
            bc_x = mat_data['bc_x'].reshape((nelx, nely)).T
            bc_y = mat_data['bc_y'].reshape((nelx, nely)).T
            F_x = mat_data['F_x'].reshape((nelx, nely)).T
            F_y = mat_data['F_y'].reshape((nelx, nely)).T
            MISES = mat_data['MISES'].reshape((nelx, nely)).T
            Real_xPhys_FEA = Real_xPhys
            bc = mat_data['bc'].astype(np.float32).squeeze()
            F = mat_data['F']

            vf_e = np.full((8192, 1), vf, dtype=np.float32).reshape((nelx, nely)).T

            # 保存为字典形式
            self.data.append({
                "Real_xPhys": torch.tensor(Real_xPhys, dtype=torch.float32).view(1, nely, nelx),
                "vf_e": torch.tensor(vf_e, dtype=torch.float32).view(1, nely, nelx),
                "bc_x": torch.tensor(bc_x, dtype=torch.float32).view(1, nely, nelx),
                "bc_y": torch.tensor(bc_y, dtype=torch.float32).view(1, nely, nelx),
                "F_x": torch.tensor(F_x, dtype=torch.float32).view(1, nely, nelx),
                "F_y": torch.tensor(F_y, dtype=torch.float32).view(1, nely, nelx),
                "MISES": torch.tensor(MISES, dtype=torch.float32).view(1, nely, nelx),
                "Real_xPhys_FEA": torch.tensor(Real_xPhys_FEA, dtype=torch.float32),
                "bc": torch.tensor(bc, dtype=torch.float32),
                "F": torch.tensor(F, dtype=torch.float32)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        condition_input = torch.cat((sample["vf_e"], sample["bc_x"], sample["bc_y"], sample["F_x"], sample["F_y"]), dim=0)
        return condition_input, sample["Real_xPhys"], sample["MISES"], sample["Real_xPhys_FEA"], sample["bc"], sample["F"]


def prepare_datasets(save_path, batch_size):
    dataset_file = os.path.join(save_path, 'processed_datasets_0.9_0.05.pt')

    # 如果存在已保存的数据集，则直接加载
    if os.path.exists(dataset_file):
        print("Loading datasets from saved file...")
        loaded_data = torch.load(dataset_file)
        train_dataset = loaded_data['train_dataset']
        val_dataset = loaded_data['val_dataset']
        test_dataset = loaded_data['test_dataset']

    else:
        # 如果不存在，进行数据加载和划分
        print("Processing and splitting dataset...")
        file_list = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith('.mat')]
        # random.shuffle(file_list)
        dataset = TopologyDataset(file_list)
        dataset_size = len(dataset)

        # 划分数据集
        train_size = int(dataset_size * 0.9)
        val_size = int(dataset_size * 0.05)
        test_size = dataset_size - train_size - val_size

        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, train_size + val_size))
        test_dataset = Subset(dataset, range(train_size + val_size, dataset_size))

        # 保存数据集到文件
        torch.save({
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }, dataset_file)
        print("Datasets have been processed and saved.")

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add skip connection
        out = self.relu(out)
        return out



class Generator(nn.Module):  # 增加池化
    def __init__(self, input_channels, condition_channels, output_channels):
        super(Generator, self).__init__()

        # 编码器部分
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels + condition_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 添加池化层，空间尺寸减半
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 添加池化层
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 添加池化层
        )

        # 解码器部分
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(96, 32, kernel_size=4, stride=2, padding=1),  # 拼接后通道数为 64 + 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(48, output_channels, kernel_size=4, stride=2, padding=1),  # 拼接后通道数为 32 + 16
            nn.Sigmoid()
        )

        # 残差模块部分
        self.resblock1 = ResidualBlock(output_channels, output_channels)
        self.resblock2 = ResidualBlock(output_channels, output_channels)

    def forward(self, noise_input, condition_input):
        # 通道拼接
        x = torch.cat((noise_input, condition_input), dim=1)

        # 编码路径（下采样）：提取特征，并保存中间结果以供解码路径拼接
        enc1 = self.encoder1(x)  # [B, 16, H/2, W/2]
        enc2 = self.encoder2(enc1)  # [B, 32, H/4, W/4]
        enc3 = self.encoder3(enc2)  # [B, 64, H/8, W/8]

        # 解码路径（上采样 + 跳跃连接）：逐步恢复空间尺寸
        dec1 = self.decoder1(enc3)  # [B, 64, H/4, W/4]
        dec2 = self.decoder2(torch.cat((dec1, enc2), dim=1))  # [B, 32, H/2, W/2]
        output = self.decoder3(torch.cat((dec2, enc1), dim=1))  # [B, output_channels, H, W]

        # 残差模块
        output = self.resblock1(output)
        output = self.resblock2(output)

        return output

class Discriminator(nn.Module):
    def __init__(self, input_channels, condition_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels + condition_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=4, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),  # 强制输出为 [B, 1, 1, 1]
            # 自适应池化将任意输入特征图缩放到指定输出大小（例如  1×1）。
            nn.Sigmoid()  # 添加 Sigmoid 将输出限制在 [0, 1]
        )

    def forward(self, input_data, condition_input):
        # 拼接输入数据和条件输入
        #  input_data：torch.Size([B, 2, 64, 128])
        x = torch.cat((input_data, condition_input), dim=1)  # [B,6, 64, 128]
        x = self.model(x)
        return x


def train_and_validate(generator, discriminator, train_loader, val_loader, optimizer_g, optimizer_d,
                       scheduler_g, scheduler_d, criterion_gan, criterion_l2, lambda_l2, lambda_vf,
                       num_epochs, data_save_path, save_model_path):
    best_val_loss = float('inf')  # 初始化为一个正无穷大值
    train_losses_g, train_losses_d, train_accuracies, val_losses_g = [], [], [], []
    val_maes, val_mses, val_re_vfs, val_re_misess, val_accuracies = [], [], [], [], []

    # 指定要保存的批次索引

    selected_batches = [0, 1, 3, 5, 7, 9, 11]  # 目标批次索引
    save_epochs = list(range(0, num_epochs, 8))  # 每隔8个epoch保存一次
    if num_epochs - 1 not in save_epochs:
        save_epochs.append(num_epochs - 1)  # 确保最后一个 epoch 保存

    for epoch in range(num_epochs):
        # 训练阶段
        generator.train()  # train() 是 torch.nn.Module 的方法,激活生成器和判别器为训练模式（train 模式）
        discriminator.train()
        train_loss_g = 0
        train_loss_d = 0
        correct_pixels_train, total_pixels_train = 0, 0  # 训练精度计数
        selected_train_samples = {'real_xPhys': [], 'fake_xPhys': [], 'epoch': []}

        for batch_idx, (condition_input, Real_xPhys, MISES, Real_xPhys_FEA, bc, F) in enumerate(train_loader):
            Fake_xPhys = generator(MISES, condition_input)

            # 判别器训练
            discriminator.zero_grad()  # 清零判别器梯度
            combined_real = torch.cat((Real_xPhys, Real_xPhys), dim=1)  # torch.Size([B, 2, 64, 128])
            combined_fake = torch.cat((Fake_xPhys.detach(), Real_xPhys), dim=1)
            # 判别器训练时，生成器的参数不更新，因此需要 detach()。首先固定 G，优化 D

            real_labels = torch.ones(combined_real.size(0), 1, 1, 1).to(
                Real_xPhys.device)  # 真标签 torch.Size([B, 1, 1, 1])
            fake_labels = torch.zeros(combined_fake.size(0), 1, 1, 1).to(Fake_xPhys.device)  # 假标签，全0张量

            # 判别器输出概率
            real_preds = discriminator(combined_real, condition_input)  # 判别器输出真实样本概率 # torch.Size([B, 1, 1, 1])
            fake_preds = discriminator(combined_fake, condition_input)  # 判别器输出生成样本概率

            # 判别器损失
            loss_d_real = criterion_gan(real_preds, real_labels)  # 真样本的对抗损失
            loss_d_fake = criterion_gan(fake_preds, fake_labels)  # 生成样本的对抗损失
            loss_d = (loss_d_real + loss_d_fake) / 2  # 判别器总损失
            loss_d.backward()  # 反向传播更新梯度，得到损失函数对每个模型参数的梯度
            optimizer_d.step()  # 更新判别器参数

            # 生成器训练
            generator.zero_grad()  # 清零梯度
            combined_fake = torch.cat((Fake_xPhys, Real_xPhys), dim=1)  # 构造生成输入

            # 更新生成器
            fake_pred = discriminator(combined_fake, condition_input)  # 判别器输出生成样本概率
            loss_g_gan = criterion_gan(fake_pred, real_labels)  # 生成器对抗损失
            loss_g_l2 = criterion_l2(Fake_xPhys, Real_xPhys)  # L2 损失
            loss_g_vf, _, _ = calculate_volume_fraction_loss(Fake_xPhys, Real_xPhys)  # 体积分数误差
            loss_g = loss_g_gan + lambda_l2 * loss_g_l2 + lambda_vf * loss_g_vf  # 总损失
            loss_g.backward()  # 梯度反向传播
            optimizer_g.step()  # 更新参数

            train_loss_g += loss_g.item()  # loss_g.item() 将张量类型的损失值转换为 Python 标量（float），+= 用于数值累加。
            train_loss_d += loss_d.item()  # train_loss_d 是所有批次中 loss_d 的累加值。

            # 计算训练像素精度
            Fake_binary = (Fake_xPhys >= 0.5).float()
            Real_binary = (Real_xPhys >= 0.5).float()
            correct_pixels_train += (Fake_binary == Real_binary).sum().item()
            total_pixels_train += Real_binary.numel()

            # 保存指定批次的第一个样本
            if batch_idx in selected_batches:
                selected_train_samples['real_xPhys'].append(Real_xPhys[0].cpu().numpy())
                selected_train_samples['fake_xPhys'].append(Fake_xPhys[0].detach().cpu().numpy())
                selected_train_samples['epoch'].append(epoch + 1)

            # 输出每个 epoch 的训练损失
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Loss D: {loss_d.item():.4f}, "  # 判别器总损失
                  f"Loss G : {loss_g.item():.4f}, "  # 生成器总损失
                  )
        train_losses_g.append(train_loss_g / len(train_loader))  # 在训练循环中记录损失
        train_losses_d.append(train_loss_d / len(train_loader))
        # 计算每个epoch的训练像素精度
        train_accuracy = correct_pixels_train / total_pixels_train
        train_accuracies.append(train_accuracy)

        # 定期保存训练集选定样本
        if epoch in save_epochs:
            np.savez(os.path.join(data_save_path, f'train_samples_epoch_{epoch + 1}.npz'),
                     real_xPhys=np.array(selected_train_samples['real_xPhys']),
                     fake_xPhys=np.array(selected_train_samples['fake_xPhys']),
                     epoch=np.array(selected_train_samples['epoch']))


        # 验证阶段
        generator.eval()  # 切换生成器到评估模式,冻结 BatchNormh和Dropout
        #  BatchNorm ，在每个小批次上对特征图进行归一化，加速模型收敛。
        #  Dropout,随机将一定比例的神经元输出置为零，减少过拟合。
        val_samples = {'real_xPhys': [], 'fake_xPhys': []}
        val_loss_g = 0  # 初始化验证损失
        val_mae, val_mse, val_re_vf, val_re_mises = 0, 0, 0, 0
        correct_pixels_val, total_pixels_val = 0, 0  # 验证精度计数

        with torch.no_grad():  # 禁用梯度计算，减少内存占用
            first_batch = True
            for  batch_idx, (condition_input, Real_xPhys, MISES, Real_xPhys_FEA, bc, F) in enumerate(val_loader):
                # 将张量转换回 NumPy 数组,用于计算全局应力
                Real_xPhys_FEA = Real_xPhys_FEA.numpy()  # (2, 64, 128)
                bc = bc.numpy()  # (2, 16770)
                F = F.numpy()  # (2, 16770, 1)
                F = F.astype(np.float64)

                Fake_xPhys = generator(MISES, condition_input)
                Fake_xPhys_FEA = Fake_xPhys.detach().squeeze(1).numpy()
                combined_fake = torch.cat((Fake_xPhys, Real_xPhys), dim=1)

                # 计算验证损失（生成器总损失）
                fake_pred = discriminator(combined_fake, condition_input)
                loss_g_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))
                loss_g_l2 = criterion_l2(Fake_xPhys, Real_xPhys)
                loss_g_vf, fake_vf, real_vf = calculate_volume_fraction_loss(Fake_xPhys, Real_xPhys)
                val_loss_g += loss_g_gan.item() + lambda_l2 * loss_g_l2 + lambda_vf * loss_g_vf

                # 保存指定批次的第一个样本
                if batch_idx in selected_batches:
                    val_samples['real_xPhys'].append(Real_xPhys[0].cpu().numpy())
                    val_samples['fake_xPhys'].append(Fake_xPhys[0].detach().cpu().numpy())

                # 计算评价指标
                mae, mse, re_vf, re_mises, accuracy = evaluate_metrics_train(
                    Fake_xPhys, Real_xPhys, fake_vf, real_vf, Real_xPhys_FEA, Fake_xPhys_FEA, bc, F, batch_size
                    # 真、假样本总体积分数
                )
                val_mae += mae
                val_mse += mse
                val_re_vf += re_vf
                val_re_mises += re_mises
                correct_pixels_val += accuracy * Real_xPhys.numel()
                total_pixels_val += Real_xPhys.numel()

        # 定期保存样本
        if epoch in save_epochs:
            np.savez(os.path.join(data_save_path, f'val_samples_epoch_{epoch + 1}.npz'),
                     real_xPhys=np.array(val_samples['real_xPhys']),
                     fake_xPhys=np.array(val_samples['fake_xPhys']))
            # real_xPhys.shape -> (3, 1, 64, 128)  # 共选择3个批次，每个批次1个样本
            # fake_xPhys.shape -> (3, 1, 64, 128)

        # 调整学习率
        scheduler_g.step(val_loss_g)
        scheduler_d.step(val_loss_g)

        # 保存模型
        if val_loss_g < best_val_loss:
            best_val_loss = val_loss_g
            torch.save(generator.state_dict(), os.path.join(save_model_path, 'ResUNet_3_best_generator.pth'))
            torch.save(discriminator.state_dict(), os.path.join(save_model_path, 'ResUNet_best_discriminator.pth'))

        # 记录精度变化,每轮的精度
        val_losses_g.append(val_loss_g / len(val_loader))
        val_maes.append(val_mae / len(val_loader))
        val_mses.append(val_mse / len(val_loader))
        val_re_vfs.append(val_re_vf / len(val_loader))
        val_re_misess.append(val_re_mises / len(val_loader))  # 确保 re_mises 是累加的结果
        val_accuracy = correct_pixels_val / total_pixels_val
        val_accuracies.append(val_accuracy)

        # 保存每轮的训练和验证损失到 npz 文件
        np.savez(os.path.join(data_save_path, 'ResUNet_training_metrics.npz'),
                 train_losses_g=train_losses_g,
                 train_losses_d=train_losses_d,
                 train_accuracies=train_accuracies,
                 val_losses_g=val_losses_g,
                 val_maes=val_maes,
                 val_mses=val_mses,
                 val_re_vfs=val_re_vfs,
                 val_re_misess=val_re_misess,
                 val_accuracies=val_accuracies)

        # 打印验证阶段的平均指标
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Val Loss G: {val_loss_g / len(val_loader):.4f},"
              f"Val MAE: {val_mae / len(val_loader):.4f},"
              f"Val MSE: {val_mse / len(val_loader):.4f},"
              f"Val RE_VF_G: {val_re_vf / len(val_loader):.4f},"
              f"Val RE_r2_G: {val_re_mises / len(val_loader):.4f}"
              f"Val Accuracy: {val_accuracy / len(val_loader):.4f}")

def test_model(generator, discriminator, test_loader, criterion_gan, criterion_l2,
               lambda_l2, lambda_vf, data_save_path, save_model_path):
    generator.load_state_dict(torch.load(os.path.join(save_model_path, 'ResUNet_3_best_generator.pth')))
    discriminator.load_state_dict(torch.load(os.path.join(save_model_path, 'ResUNet_best_discriminator.pth')))
    generator.eval()  # 切换到评估模式，冻结 BN 和 Dropout
    discriminator.eval()

    test_loss_g = 0
    test_mae, test_mse, test_re_vf, test_re_mises, test_accuracy , test_r2_vf= 0, 0, 0, 0, 0, 0
    mae_list = []  # 存储每个样本的 mae
    mse_list = []  # 存储每个样本的 mse
    re_vf_list = []  # 存储每个样本的 re_vf
    re_mises_list = []  # 存储每个样本的 re_mises
    real_xPhys_list = []  # 保存 Real_xPhys
    fake_xPhys_list = []  # 保存 Fake_xPhys
    accuracy_list = []  # accuracy
    real_vm_list = []  # 保存 Real_xPhys
    fake_vm_list = []  # 保存 Fake_xPhys
    r2_vf_list = []  # 保存 r2_vf

    with torch.no_grad():  # 禁用梯度计算
        for condition_input, Real_xPhys, MISES, Real_xPhys_FEA, bc, F in test_loader:
            Real_xPhys_FEA = Real_xPhys_FEA.numpy()
            bc = bc.numpy()
            F = F.numpy().astype(np.float64)

            # 生成伪拓扑分布
            Fake_xPhys = generator(MISES, condition_input)
            Fake_xPhys_FEA = Fake_xPhys.detach().squeeze(1).numpy()
            combined_fake = torch.cat((Fake_xPhys, Real_xPhys), dim=1)

            # 判别器预测
            fake_pred = discriminator(combined_fake, condition_input)
            loss_g_gan = criterion_gan(fake_pred, torch.ones_like(fake_pred))
            loss_g_l2 = criterion_l2(Fake_xPhys, Real_xPhys)
            loss_g_vf, fake_vf, real_vf = calculate_volume_fraction_loss(Fake_xPhys, Real_xPhys)

            test_loss_g += loss_g_gan.item() + lambda_l2 * loss_g_l2 + lambda_vf * loss_g_vf

            # 计算评价指标
            mae, mse, re_vf, re_mises, accuracy , r2_vf ,Real_vm, Fake_vm  = evaluate_metrics(
                Fake_xPhys, Real_xPhys, fake_vf, real_vf, Real_xPhys_FEA, Fake_xPhys_FEA, bc, F, batch_size
            )
            test_mae += mae
            test_mse += mse
            test_re_vf += re_vf
            test_re_mises += re_mises
            test_accuracy += accuracy
            test_r2_vf += r2_vf

            mae_list.append(mae)
            mse_list.append(mse)
            re_vf_list.append(re_vf)
            re_mises_list.append(re_mises)
            real_xPhys_list.append(Real_xPhys.cpu().numpy())
            fake_xPhys_list.append(Fake_xPhys.cpu().numpy())
            accuracy_list.append(accuracy)
            r2_vf_list.append(r2_vf)
            real_vm_list.append(Real_vm.cpu().numpy())
            fake_vm_list.append(Fake_vm.cpu().numpy())

    # 保存测试集数据
    mae_array = np.array(mae_list)
    mse_array = np.array(mse_list)
    re_vf_array = np.array(re_vf_list)
    r2_vf_array = np.array(r2_vf_list)
    accuracy_array = np.array(accuracy_list)
    re_mises_array = np.array(re_mises_list)
    np.savez(os.path.join(data_save_path, 'ResUNet_3_test_summary.npz'),
             test_mae=test_mae / len(test_loader),
             test_mse=test_mse / len(test_loader),
             test_re_vf=test_re_vf / len(test_loader),
             test_re_vm=test_re_mises / len(test_loader),
             test_accuracy=test_accuracy / len(test_loader),
             test_r2_vf=test_r2_vf / len(test_loader),
             test_mse_all=mse_array,
             test_mae_all=mae_array,
             test_re_vf_all=re_vf_array,
             test_re_vm_all=re_mises_array,
             test_accuracy_all=accuracy_array,
             test_r2_vf_all=r2_vf_array,
             real_xPhys=np.array(real_xPhys_list),
             fake_xPhys=np.array(fake_xPhys_list),
             real_vm=np.array(real_vm_list),
             fake_vm=np.array(fake_vm_list))

    print(f"Test MAE: {test_mae / len(test_loader):.4f}")
    print(f"Test MSE: {test_mse / len(test_loader):.4f}")
    print(f"Test RE_VF_G: {test_re_vf / len(test_loader):.4f}")
    print(f"Test RE_VM_G: {test_re_mises / len(test_loader):.4f}")
    print(f"Test accuracy: {test_accuracy / len(test_loader):.4f}")
    print(f"Test r2_vf: {test_r2_vf / len(test_loader):.4f}")




def test_specific_case(generator, file_path, save_model_path):
    generator.eval()
    nelx, nely = 128, 64  # 拓扑优化尺寸
    mat_data = sio.loadmat(file_path)
    Real_xPhys = mat_data['Real_xPhys'].reshape((nelx, nely)).T  # (64, 128)
    vf = mat_data['vf'].item()  # scalar
    bc_x = mat_data['bc_x'].reshape((nelx, nely)).T  # (64, 128)
    bc_y = mat_data['bc_y'].reshape((nelx, nely)).T  # (64, 128)
    F_x = mat_data['F_x'].reshape((nelx, nely)).T  # (64, 128)
    F_y = mat_data['F_y'].reshape((nelx, nely)).T  # (64, 128)
    MISES = mat_data['MISES'].reshape((nelx, nely)).T  # (64, 128)
    Real_xPhys_FEA = Real_xPhys  # (64, 128)
    bc = mat_data['bc'].astype(np.float32).squeeze()  # (16770,)
    F = mat_data['F'].astype(np.float32)  # (16770, 1)

    # Calculate vf_e (volume fraction expansion)
    vf_e = np.full((8192, 1), vf, dtype=np.float32).reshape((nelx, nely)).T  # (64, 128)

    # Convert to PyTorch tensors and adjust dimensions (adding batch dimension)
    Real_xPhys = torch.tensor(Real_xPhys, dtype=torch.float32).unsqueeze(0)  # (1, 64, 128)
    vf_e = torch.tensor(vf_e, dtype=torch.float32).unsqueeze(0)  # (1, 64, 128)
    bc_x = torch.tensor(bc_x, dtype=torch.float32).unsqueeze(0)  # (1, 64, 128)
    bc_y = torch.tensor(bc_y, dtype=torch.float32).unsqueeze(0)  # (1, 64, 128)
    F_x = torch.tensor(F_x, dtype=torch.float32).unsqueeze(0)  # (1, 64, 128)
    F_y = torch.tensor(F_y, dtype=torch.float32).unsqueeze(0)  # (1, 64, 128)
    MISES = torch.tensor(MISES, dtype=torch.float32).unsqueeze(0)  # (1, 64, 128)
    Real_xPhys_FEA = torch.tensor(Real_xPhys_FEA, dtype=torch.float32).unsqueeze(0)  # (1, 64, 128)
    bc = torch.tensor(bc, dtype=torch.float32)  # (16770,)
    F = torch.tensor(F, dtype=torch.float32)  # (16770, 1)

    # Organize the condition input as it was done during training
    condition_input = torch.cat((
        vf_e.view(1, nely, nelx),
        bc_x.view(1, nely, nelx),
        bc_y.view(1, nely, nelx),
        F_x.view(1, nely, nelx),
        F_y.view(1, nely, nelx)
    ), dim=0).unsqueeze(0)  # (1, 5, 64, 128)

    MISES = MISES.unsqueeze(0) # (1, 1, 64, 128)
    Real_xPhys_FEA = Real_xPhys_FEA.numpy()  # ndarray(1, 64, 128)
    bc = bc.numpy()
    F = F.numpy().astype(np.float64)

    # 加载最优模型数据
    generator.load_state_dict(torch.load(os.path.join(save_model_path, 'ResUNet_best_generator.pth')))
    with torch.no_grad():  # 关闭梯度计算，提高效率
        Fake_xPhys = generator(MISES, condition_input)  # (1, 1, 64, 128)

    #  **转换生成结果**

    fake_xPhys_numpy = Fake_xPhys.squeeze().detach().cpu().numpy() # (64, 128)
    Fake_xPhys_FEA = np.expand_dims(fake_xPhys_numpy, axis=0)

    _,Real_vonmises_distribution = stress_distribution(Real_xPhys_FEA, F, bc)
    _, Fake_vonmises_distribution = stress_distribution(Fake_xPhys_FEA, F, bc)

    # **可视化**
    Real_vonmises_reshaped = Real_vonmises_distribution.numpy().reshape(64, 128)
    Fake_vonmises_reshaped = Fake_vonmises_distribution.numpy().reshape(64, 128)

    # 创建一个大画布
    plt.figure(figsize=(15, 15))

    # 第一个子图：绘制真实拓扑与生成拓扑
    plt.subplot(2, 2, 1)  # 1 行 2 列的第一个子图
    plt.imshow(Real_xPhys.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Truth Topology")
    plt.colorbar()

    # 第一个子图：绘制生成拓扑结构
    binary_fake_xPhys = fake_xPhys_numpy > 0.5
    nely, nelx = binary_fake_xPhys.shape
    X, Y = np.meshgrid(np.arange(1, nelx + 1), np.arange(1, nely + 1))
    plt.subplot(2, 2, 2)  # 1 行 2 列的第二个子图
    contour = plt.contour(X, Y, fake_xPhys_numpy, levels=[0.5], colors='black', linewidths=2)  # 只绘制等值线
    plt.imshow(binary_fake_xPhys, cmap='gray', origin='lower', alpha=0.5)  # 背景为白色和黑色
    plt.title('Generated Topology')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 第二个子图：绘制 von Mises 应力分布
    plt.subplot(2, 2, 3)  # 1 行 2 列的第二个子图
    plt.imshow(Real_vonmises_reshaped, cmap='jet', aspect='auto')
    plt.colorbar(label="Von Mises Stress")  # 显示应力颜色条
    plt.title("Real Stress Distribution")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(2, 2, 4)  # 1 行 2 列的第二个子图
    plt.imshow(Fake_vonmises_reshaped, cmap='jet', aspect='auto')
    plt.colorbar(label="Von Mises Stress")  # 显示应力颜色条
    plt.title("Fake Stress Distribution")
    plt.xlabel("x")
    plt.ylabel("y")

    # 展示所有图像
    plt.show()
    return Fake_xPhys


if __name__ == "__main__":
    start_time = time.time()
    # 数据加载
    data_save_path = r"E:\work3\DPTO\GAN_TO code\GAN_to\dataset_single\paper_data\ResUNet"
    save_path = r"E:\work3\DPTO\GAN_TO code\GAN_to\dataset_single\dataset_all"  # 样本加载
    save_model_path = r"E:\work3\DPTO\GAN_TO code\GAN_to\dataset_single\paper_data\ResUNet"

    # # 创建 DataLoader
    batch_size = 1   # 64
    train_loader, val_loader, test_loader = prepare_datasets(save_path, batch_size)

    # 实例化生成器和判别器
    generator = Generator(input_channels=1, condition_channels=5, output_channels=1)
    discriminator = Discriminator(input_channels=2, condition_channels=5)  # Fake_xPhys + Real_xPhys

    # 损失函数和优化器,最小化生成器损失函数，最大化判别器损失函数
    criterion_gan = nn.BCELoss()  # 对抗损失(二值交叉熵损失)，BCELoss 主要用于判别器的二分类任务。
    criterion_l2 = nn.MSELoss()   # L2即均方误差 (MSE)，用于衡量生成结果和真实数据的逐点差异，对大误差 敏感（惩罚大误差更强），但可能受离群点影响。
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.99))  # 动量超参数，0.5历史梯度的权重。 0.999：平方梯度的权重。
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.99))

    # 超参数
    lambda_l2 = 10000
    lambda_vf = 1000
    num_epochs = 65  # 训练的轮数

    # 定义调度器
    scheduler_g = lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.1, patience=10, verbose=True)
    scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.1, patience=10, verbose=True)

    # 训练及验证循环
    train_and_validate(generator, discriminator, train_loader, val_loader,
                       optimizer_g, optimizer_d, scheduler_g, scheduler_d, criterion_gan, criterion_l2,
                       lambda_l2, lambda_vf, num_epochs, data_save_path, save_model_path)
    end_time = time.time()
    # 计算运行时间（转换为分钟）
    elapsed_time = (end_time - start_time) / 60
    print(f"Total training and validation time: {elapsed_time:.2f} minutes")

    # # # 模型测试阶段
    test_model(generator, discriminator, test_loader,
               criterion_gan, criterion_l2, lambda_l2, lambda_vf, data_save_path, save_model_path)

