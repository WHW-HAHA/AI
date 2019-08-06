
# coding: utf-8

# # 你的第一个神经网络
# 
# 在此项目中，你将构建你的第一个神经网络，并用该网络预测每日自行车租客人数。我们提供了一些代码，但是需要你来实现神经网络（大部分内容）。提交此项目后，欢迎进一步探索该数据和模型。

# In[24]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time


# ## 加载和准备数据
# 
# 构建神经网络的关键一步是正确地准备数据。不同尺度级别的变量使网络难以高效地掌握正确的权重。我们在下方已经提供了加载和准备数据的代码。你很快将进一步学习这些代码！

# In[25]:

data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)


# In[26]:

rides.head()


# ## 数据简介
# 
# 此数据集包含的是从 2011 年 1 月 1 日到 2012 年 12 月 31 日期间每天每小时的骑车人数。骑车用户分成临时用户和注册用户，cnt 列是骑车用户数汇总列。你可以在上方看到前几行数据。
# 
# 下图展示的是数据集中前 10 天左右的骑车人数（某些天不一定是 24 个条目，所以不是精确的 10 天）。你可以在这里看到每小时租金。这些数据很复杂！周末的骑行人数少些，工作日上下班期间是骑行高峰期。我们还可以从上方的数据中看到温度、湿度和风速信息，所有这些信息都会影响骑行人数。你需要用你的模型展示所有这些数据。

# In[27]:

rides[:24*10].plot(x='dteday', y='cnt')
type(rides)


# ### 虚拟变量（哑变量）
# 
# 下面是一些分类变量，例如季节、天气、月份。要在我们的模型中包含这些数据，我们需要创建二进制虚拟变量。用 Pandas 库中的 `get_dummies()` 就可以轻松实现。

# In[28]:

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()


# ### 调整目标变量
# 
# 为了更轻松地训练网络，我们将对每个连续变量标准化，即转换和调整变量，使它们的均值为 0，标准差为 1。
# 
# 我们会保存换算因子，以便当我们使用网络进行预测时可以还原数据。

# In[29]:

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    print('mean: ',mean,'std: ',std)
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# ### 将数据拆分为训练、测试和验证数据集
# 
# 我们将大约最后 21 天的数据保存为测试数据集，这些数据集会在训练完网络后使用。我们将使用该数据集进行预测，并与实际的骑行人数进行对比。

# In[30]:

# Save data for approximately the last 21 days 
test_data = data[-21*24:]

# Now remove the test data from the data set 
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]


# 我们将数据拆分为两个数据集，一个用作训练，一个在网络训练完后用来验证网络。因为数据是有时间序列特性的，所以我们用历史数据进行训练，然后尝试预测未来数据（验证数据集）。

# In[31]:

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
print(train_features.shape)
val_features, val_targets = features[-60*24:], targets[-60*24:]
print(val_features.shape)


# ## 开始构建网络
# 
# 下面你将构建自己的网络。我们已经构建好结构和反向传递部分。你将实现网络的前向传递部分。还需要设置超参数：学习速率、隐藏单元的数量，以及训练传递数量。
# 
# <img src="assets/neural_network.png" width=300px>
# 
# 该网络有两个层级，一个隐藏层和一个输出层。隐藏层级将使用 S 型函数作为激活函数。输出层只有一个节点，用于递归，节点的输出和节点的输入相同。即激活函数是 $f(x)=x$。这种函数获得输入信号，并生成输出信号，但是会考虑阈值，称为激活函数。我们完成网络的每个层级，并计算每个神经元的输出。一个层级的所有输出变成下一层级神经元的输入。这一流程叫做前向传播（forward propagation）。
# 
# 我们在神经网络中使用权重将信号从输入层传播到输出层。我们还使用权重将错误从输出层传播回网络，以便更新权重。这叫做反向传播（backpropagation）。
# 
# > **提示**：你需要为反向传播实现计算输出激活函数 ($f(x) = x$) 的导数。如果你不熟悉微积分，其实该函数就等同于等式 $y = x$。该等式的斜率是多少？也就是导数 $f(x)$。
# 
# 
# 你需要完成以下任务：
# 
# 1. 实现 S 型激活函数。将 `__init__` 中的 `self.activation_function`  设为你的 S 型函数。
# 2. 在 `train` 方法中实现前向传递。
# 3. 在 `train` 方法中实现反向传播算法，包括计算输出错误。
# 4. 在 `run` 方法中实现前向传递。
# 
#   

# In[32]:

class NeuralNetwork():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
#         ct = time.time()
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_to_hidden = np.random.normal(0, self.input_nodes**-0.5,
                                                        (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0, self.hidden_nodes**-0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        self.activation_function = lambda x: 1/(1+np.exp(-x))
#         print('Initime1',time.time()-ct)

    def train(self, features, targets):
#         ct = time.time()
        features = np.array(features, ndmin= 3)
        targets = np.array(targets, ndmin= 3)

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = final_inputs

            self.error = y - final_outputs
            error_term = self.error

            hidden_error = np.dot(self.weights_hidden_to_output, error_term) # shape should be hidden_nodes x 1 --> 16 x 1 or 2 x 1
            hidden_error_term = hidden_error * hidden_outputs.T * (1-hidden_outputs).T # 2 x 1

            delta_weights_h_o += error_term * hidden_outputs.T  # shape should like self.w_h_o --> 1 x 2 or 1 x 16
            # delta_weights_i_h += hidden_error_term * X        # shape should like self.w_i_h --> 2 x 3 or 16 x 56
            delta_weights_i_h += X.T * hidden_error_term.T

        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
#         print('per loop time1', (time.time()-ct)*100)
    def run(self, features):
#         c_time = time.time()
        features = np.array(features, ndmin=2)
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs  = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs
#         print('Run time:', time.time()-c_time)

        return final_outputs.T


# In[34]:

def MSE(y, Y):
    return np.mean((y-Y)**2)


# ## 单元测试
# 
# 运行这些单元测试，检查你的网络实现是否正确。这样可以帮助你确保网络已正确实现，然后再开始训练网络。这些测试必须成功才能通过此项目。

# In[35]:

import unittest

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328], 
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, -0.20185996], 
                                              [0.39775194, 0.50074398], 
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)


# ## 训练网络
# 
# 现在你将设置网络的超参数。策略是设置的超参数使训练集上的错误很小但是数据不会过拟合。如果网络训练时间太长，或者有太多的隐藏节点，可能就会过于针对特定训练集，无法泛化到验证数据集。即当训练集的损失降低时，验证集的损失将开始增大。
# 
# 你还将采用随机梯度下降 (SGD) 方法训练网络。对于每次训练，都获取随机样本数据，而不是整个数据集。与普通梯度下降相比，训练次数要更多，但是每次时间更短。这样的话，网络训练效率更高。稍后你将详细了解 SGD。
# 
# 
# ### 选择迭代次数
# 
# 也就是训练网络时从训练数据中抽样的批次数量。迭代次数越多，模型就与数据越拟合。但是，如果迭代次数太多，模型就无法很好地泛化到其他数据，这叫做过拟合。你需要选择一个使训练损失很低并且验证损失保持中等水平的数字。当你开始过拟合时，你会发现训练损失继续下降，但是验证损失开始上升。
# 
# ### 选择学习速率
# 
# 速率可以调整权重更新幅度。如果速率太大，权重就会太大，导致网络无法与数据相拟合。建议从 0.1 开始。如果网络在与数据拟合时遇到问题，尝试降低学习速率。注意，学习速率越低，权重更新的步长就越小，神经网络收敛的时间就越长。
# 
# 
# ### 选择隐藏节点数量
# 
# 隐藏节点越多，模型的预测结果就越准确。尝试不同的隐藏节点的数量，看看对性能有何影响。你可以查看损失字典，寻找网络性能指标。如果隐藏单元的数量太少，那么模型就没有足够的空间进行学习，如果太多，则学习方向就有太多的选择。选择隐藏单元数量的技巧在于找到合适的平衡点。

# In[36]:

import sys
# from NN import NeuralNetwork
### TODO:Set the hyperparameters here, you need to change the defalut to get a better solution ###
import time
iterations = 1000
learning_rate = 0.8
hidden_nodes = 2
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(iterations):
    if (e > 500):
        network.lr = 0.1
    if (e> 800):
        network.lr = 0.01
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values, 
                              train_targets.ix[batch]['cnt']):
#         c_time= time.time()
        network.train(record, target)
#         print('train time: ', str(time.time()-c_time))
    # Printing out the training progress
    c_time = time.time()
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
#     print('Train_loss_cal: ', time.time()-c_time)
    c_time = time.time()
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
#     print('Val_loss_cal: ', time.time()-c_time)
    sys.stdout.write("\rProgress: " + str(100 * e/float(iterations))[:4]                     + "% ... Training loss: " + str(train_loss)[:5]                     + " ... Validation loss: " + str(val_loss)[:5]                     + "...other Time" + str(time.time()-c_time))
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


# In[37]:

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()


# ## 检查预测结果
# 
# 使用测试数据看看网络对数据建模的效果如何。如果完全错了，请确保网络中的每步都正确实现。

# In[38]:

fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']

predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)


# ## 可选：思考下你的结果（我们不会评估这道题的答案）
# 
#  
# 请针对你的结果回答以下问题。模型对数据的预测效果如何？哪里出现问题了？为何出现问题呢？
# 
# > **注意**：你可以通过双击该单元编辑文本。如果想要预览文本，请按 Control + Enter
# 
# #### 请将你的答案填写在下方
# 
# #### 抛开模型的调试和超参数，测试数据集中，前半段中并没有节假日，这使得预测的准确度相对更高，预测的波动更小。
# #### 同时验证模型中仅仅使用了‘cnt’这一列数据作为验证数据，所以结果也并不完全准确
