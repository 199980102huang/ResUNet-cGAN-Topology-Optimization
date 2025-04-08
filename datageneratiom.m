%% 建立样本循环
clc,clear,close all;

%% 初始化参数
% 修改起始样本编号
start_sample =1; % 从第1个样本开始
num_samples = 10000; % 样本数量
nelx = 128; nely = 64; 

tolne = nelx*nely; % 单元数量
tolnd = (nelx+1)*(nely+1); % 节点数量
ndof = 2*tolnd; % 自由度数量

%% 开始计时
tic; % 开始计时

for i = start_sample:num_samples
    %% 随机生成输入参数
    % 体积分数
    vf = 0.2 + 0.3 * rand; % 在 [0.2, 0.5] 范围内随机选择

    % 边界条件
    min_fixed_nodes = 2; % 最少固定节点数量
    max_fixed_nodes = 2*(nely+1); % 最多固定节点数量
    num_fixed_nodes = randi([min_fixed_nodes, max_fixed_nodes]); %  生成一个随机整数，范围在 [2, 2*(nely+1)]
    fixeddofs = sort(randperm(ndof, num_fixed_nodes)); %用于从 1 到 ndof 的范围内，随机抽取num_fixed_nodes个不重复的整数
    bc = zeros(ndof, 1);
    bc(fixeddofs) = 1; % 设置固定节点

    % 载荷
    F = zeros(ndof, 1);
    load_position = randi([1, ndof], 1); % 随机加载一个节点
    load_value = -1 + 2 * rand ; % 随机加载大小
    F(load_position) = load_value;

    %% 映射到单元数据
    % 初始化单元数据
    bc_x = zeros(tolne, 1); % 单元 x 方向边界条件
    bc_y = zeros(tolne, 1); % 单元 y 方向边界条件
    F_x = zeros(tolne, 1);  % 单元 x 方向载荷
    F_y = zeros(tolne, 1);  % 单元 y 方向载荷

    % 遍历单元
    for elx = 1:nelx
        for ely = 1:nely
            % 当前单元编号
            elem_id = (elx-1) * nely + ely;

            % 节点编号
            n1 = (nely+1) * (elx-1) + ely; % 左上角节点编号
            n2 = (nely+1) * elx + ely;     % 右上角节点编号

            % 获取当前单元的 8 个自由度编号
            dofs = [...
                2*n1-1, 2*n1;             % 左上角节点
                2*n1+1, 2*n1+2;           % 左下角节点
                2*n2-1, 2*n2;             % 右上角节点
                2*n2+1, 2*n2+2            % 右下角节点
            ];

            % 提取单元的 x 和 y 方向边界条件和载荷
            bc_x(elem_id) = max(bc(dofs(1:2:end)));  % 提取 x 方向边界条件
            bc_y(elem_id) = max(bc(dofs(2:2:end)));  % 提取 y 方向边界条件
            F_x(elem_id) = sum(F(dofs(1:2:end)));    % 汇总 x 方向载荷
            F_y(elem_id) = sum(F(dofs(2:2:end)));    % 汇总 y 方向载荷
        end
    end


    %% 调用拓扑优化代码
    [xPhys,MISES1] = topo_single(nelx, nely, vf,fixeddofs,F);

    %% 保存当前样本
    data.loop = i;
    data.Real_xPhys = xPhys; % (8192×1, 1)
    data.vf = vf; % (1×1, 1)
    data.bc = bc; % (16770×1, 1)
    data.bc_x = bc_x; % (8192×1, 1)
    data.bc_y = bc_y;  % (8192×1, 1)
    data.F = F; % (16770×1, 1)
    data.F_x = F_x;   % (8192×1, 1)
    data.F_y = F_y;   % (8192×1, 1)
    data.MISES = MISES1;  % (8192×1, 1)
%     data.MISES_end = MISES;  % (8192×1, 1)
%     data.Real_obj = Real_obj;  % (1, 1)

    % 保存固定节点和载荷信息
%     data.fixed_dofs = fixeddofs; % 保存固定节点索引
%     data.load_position = load_position; % 保存加载的位置索引
%     data.load_value = load_value; % 保存加载的大小

     % 创建完整的保存文件路径
    save_path = 'E:\work3\DPTO\GAN_TO code\GAN_to\dataset_single\dataset_all';
    filename = sprintf('%s\\data_sample_%04d.mat', save_path, i);
    save(filename, '-struct', 'data');
end

%% 结束计时并输出时间
elapsed_time = toc; % 结束计时
fprintf('程序运行完成，共耗时 %.2f 秒。\n', elapsed_time);
%% 定义边界条件（L型梁） %%
% vf = 0.3;
% F = zeros(ndof, 1); % 
% F(ndof - nely + 2) = -1; % 
% fixeddofs = [1:2*(nely+1):ndof,2:2*(nely+1):ndof]; 
% freedofs = setdiff(1:2*tolnd,fixeddofs);
% bc = zeros(ndof, 1); % 初始化边界条件向量
% bc(fixeddofs) = 1;   % 对于约束节点，设置为 1
% passive = zeros(nely,nelx);
% passive(1:nely/2,nelx/2+1:end) = 1;
% 
% topo_single(nelx, nely, vf,fixeddofs,F,passive);

