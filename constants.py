# 设置模型大小
n = 1
dim = 2**n
# 设置初始参数
p0 = 0  # 初始噪声强度
pT = 1  # 最终噪声强度
# 加噪层数
pre_layers = 10
# 加噪分裂路径数
pre_Kraus_PERLAYES = 4

# 解躁层数
LAYERS = 1
# 解躁分裂数
Kraus_PERLAYES = 10

Lambda = 0.02
lambda_l = [1]
lambda_l.extend([Lambda] * (pre_layers - 1))

filepath = fr"D:\pycharm_projects\HQMM\qstate_QHMNN_Depolarizing\qstate_QHMNN_Depolarizing_parallel2\Depolarizing_output\Qubit_{n}\{pre_layers}_{pre_Kraus_PERLAYES}_{LAYERS}_{Kraus_PERLAYES}\{Lambda}"
# 设置轮次
Epochs = 500
Epochs_constrained = 1000
# 设置batch, 最好为2的幂数
batch_train = 70
batch_test = 10
# 设置分类任务数
CLASS_task = 2
# 几分类任务
CLASS = 2
# 训练集个数
train_num = 2000
# 测试集个数0.9
test_num = 1000
# 编码方式
encoding_method = 'amplitude_code'
# 迭代次数0.92
maxIter = 1
tau_perlayer = 0.2  # 学习率, 0~1
tau_train_all = 0.2  # 学习率, 0~1
# 学习率衰减率, 0~1
alpha = 0.01
# 动量记忆因子, 0~1
beta = 0.99
class_num = 2
epsilon = 0.05  # 扰动参数
d = 2
