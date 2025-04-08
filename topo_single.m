%% 米塞斯应力为目标，体积为约束的代码，用SIMP方法实现应力最小化

function [xPhys,MISES1] = topo_single(nelx, nely, vf,fixeddofs,F)

%% 定义参数 %%
rmin = 3;
E0 = 1; Emin = 1e-9 * E0; nu = 0.3;
tolne = nelx*nely; tolnd = (nelx+1)*(nely+1);
tolvol = tolne; ndof = 2*tolnd;

%% 准备有限元分析 %%
nodenrs = reshape(1:tolnd,1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,tolne,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1], tolne,1);
iK = reshape(kron(edofMat,ones(8,1))', 64*tolne,1);
jK = reshape(kron(edofMat,ones(1,8))', 64*tolne,1);
A11 = [12 3 -6 -3; 3 12 3 0; -6 3 12 -3; -3 0 -3 12];
A12 = [-6 -3 0 3; -3 -6 -3 -6; 0 -3 -6 3; 3 -6 3 -6];
B11 = [-4 3 -2 9; 3 -4 -9 4; -2 -9 -4 -3; 9 4 -3 -4];
B12 = [ 2 -3 4 -9; -3 2 9 -2; 4 9 2 3; -9 -2 3 2];
KE0 = 1/(1-nu^2)/24*([A11 A12; A12' A11]+nu*[B11 B12; B12' B11]);


%% 多样本
freedofs = setdiff(1:2*tolnd,fixeddofs);

%% 过滤器预处理
iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1); jH = ones(size(iH)); sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                k = k+1;
                iH(k) = e1;
                jH(k) = e2;
                sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH,jH,sH); Hs = sum(H,2);

%% 初始化设计变量 %%
beta = 0; penal = 3;
x = repmat(vf,nely,nelx);
q = 0.5; % 应力松弛系数，避免应力奇异
p = 8; % P-norm凝聚参数

%% 初始化迭代 %%
loop = 0; obj = 0.; neig = length(x(:));
change = 1.; ichange = 1; n = neig;
xmin = 0*ones(n,1); xmax = 1*ones(n,1);
low = xmin; upp = xmax;
xold1 = x(:);  xold2 = x(:); clf;
Obj = []; Volf = [];
MISES1 = []; 
MISES = []; 
% cArray=zeros(200,2);

%% 开始迭代 %%
while (loop < 80 || change >= 0.005 || beta < 20)
    loop = loop + 1; objold = obj;
    if loop > 150
        
       break;
    end
    
    %% 投影到密度场 %%
    xTlide = (H*x(:))./Hs;
    [xPhys,edproj] = THRESHOLD2(xTlide,beta);
%     edproj(find(passive)) = 0;
%     xPhys(find(passive)) = 0;

    %% 做有限元分析 %%
    sK = reshape(KE0(:)*(Emin+xPhys'.^penal*(E0-Emin)), 64*tolne, 1);
    K = sparse(iK,jK,sK); K=(K+K')/2;
    U = zeros(ndof,1);
    U(freedofs,:) = K(freedofs,freedofs)\F(freedofs,:);

    %% MISES应力计算 %%   %% 设置应力分析
    L = 0.01;
    B = (1/2/L)*[-1 0 1 0 1 0 -1 0; 0 -1 0 -1 0 1 0 1; -1 -1 -1 1 1 1 1 -1];
    D = (1/(1-nu^2))*[1 nu 0; nu 1 0; 0 0 (1-nu)/2];
    S = (U(edofMat)*(D*B)').*repmat(xPhys.^q,1,3);
    MISES = reshape(sqrt(sum(S.^2,2)-S(:,1).*S(:,2)+2.*S(:,3).^2),nely,nelx);
    MISES = MISES(:);

    %% 保存第一次迭代的 MISES %%
    if loop == 5
        MISES1 = reshape(MISES, nely*nelx,1); % 保存第一次迭代的 MISES
    end
%     MISES(find(passive)) = 0;

    %% 计算目标函数（P-norm凝聚） %%
    obj = (sum(MISES.^p))^(1/p);

    %% 目标函数灵敏度评估 %%
    DvmDs = zeros(tolne,3); % word文档里面的b
    dpn_dvms = (sum(MISES.^p))^(1/p-1);
    index_matrix = edofMat';
    for i = 1:tolne
        DvmDs(i,1) = 1/2/MISES(i) * (2*S(i,1)-S(i,2));
        DvmDs(i,2) = 1/2/MISES(i) * (2*S(i,2)-S(i,1));
        DvmDs(i,3) = 3/MISES(i) * S(i,3);
    end
%     DvmDs(find(passive),:) = 0;
    temp = zeros(tolne,1);
    for i = 1:tolne
        u = reshape(U(edofMat(i,:),:)',[],1); % 每个单元的位移
        temp(i) = q*(xPhys(i))^(q-1)*MISES(i)^(p-1)*DvmDs(i,:)*D*B*u;
    end
%     temp(find(passive)) = 0;
    T1 = dpn_dvms*temp; %% 灵敏度的第一部分（表达式没有问题）
    gama = zeros(ndof,1); %% 伴随方程的右端项
    for i = 1:tolne
        index = index_matrix(:,i);
        gama(index) = gama(index)+xPhys(i)^q*dpn_dvms*B'*D'*DvmDs(i,:)'*MISES(i).^(p-1);
    end
    lamda = zeros(ndof,1);
    % 第二部分伴随方程求解
    lamda(freedofs,:) = K(freedofs,freedofs)\gama(freedofs,:);
    T2 = zeros(tolne,1);
    for i = 1:tolne %% 灵敏度的第二部分
        index = index_matrix(:,i);
        T2(i) = -lamda(index)'*penal*xPhys(i)^(penal-1)*KE0*U(index);
    end
    ce = T1 + T2;
%     ce(find(passive)) = 0;
    
    dcdx = H*(ce(:).*edproj(:)./Hs);
    vol = sum(xPhys(:));
    voldgdx = H*(edproj(:)./Hs);

    %% 更新设计变量 %%
    m = 1;
    cc = 10000*ones(m,1); d = zeros(m,1); a0 = 1; a = zeros(m,1);
    fval = zeros(m, 1); dfdx = zeros(m, n);
    fval(1) = 100*(vol/tolvol-vf);
    dfdx(1,:) = 100*voldgdx/tolvol;
%     xmax = min(1, x(:) + 0.1); xmin = max(x(:) - 0.1,0);
    [xmma,ymma,zmma,lam,xsi,eta,mu,zet,S,low,upp]=...
        mmasub(m,n,loop,x(:),xmin,xmax,xold1,xold2, ...
        obj,dcdx/10^2,fval,dfdx,low,upp,a0,a,cc,d);
    xold2 = xold1; xold1 = x(:); x = xmma;
    
    %% 投影参数更新 %%
    change = abs(obj-objold)/obj;
    if change < 0.005 && loop > 30
        ichange = ichange+1;
    else
        ichange = 1;
    end
    if mod(ichange,3) == 0
        beta = min(beta + 1.5,20);
    end
    
    %% 输出结果并绘制密度图 %%
      disp ([' It.: ' sprintf('%5i',loop)...
             ' Obj.: ' sprintf('%10.4f',obj) ...
             ' Vol.: ' sprintf('%7.4f',vol/tolvol)...
             ' numdesvars.: ' sprintf('%5i',neig)...
             ' beta.: ' sprintf('%5.1f',beta)...
             ' ch.: ' sprintf('%6.3f\n',change )])
      figure(1); clf;
      set(gcf, 'color', 'w');
      displayx = reshape(xPhys, nely, nelx);
      colormap(gray); clims=[-1 0];imagesc(-displayx,clims); 
      axis equal; axis tight;
      title('Elemental density');
      set(gca,'XTick',[0 1e5]);set(gca,'YTick',[0 1e5]);
      axis on; % 隐藏坐标轴和边框

      figure(2); clf; %% 应力图
      set(gcf, 'color', 'w');
      colormap('jet'), imagesc(reshape(MISES,nely,nelx)); axis equal tight;
      title('Stress distribution');
      set(gca,'XTick',[0 1e5]);set(gca,'YTick',[0 1e5]);
      axis on; % 隐藏坐标轴和边框
end

% save MISES; % 应力矩阵 (nelx*nely,1)
% save volfrac;   % 体积分数矩阵 (1,1)
% save bc;   % 边界条件 (ndof, 1)
% save F;          % 载荷条件  (ndof, 1)
% save display;  % 灰度图(nelx, nely)

end
