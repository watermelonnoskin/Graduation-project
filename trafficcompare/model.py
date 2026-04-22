import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
import math

# update-decay如何实现展示
class Evolution(Layer):

    def __init__(self, dr2, use_smooth_gate=True, **kwargs):
        # 定义维度，也就是ppt里讲的2D
        self.dr2 = dr2
        self.use_smooth_gate = bool(use_smooth_gate)
        # 继承父类的属性和方法
        super(Evolution, self).__init__(**kwargs)
    #这就是权重矩阵
    def build(self, input_shape):
        self.w1 = self.add_weight(name='wl',
                                  shape=(2 * self.dr2, self.dr2),
                                  initializer=keras.initializers.RandomNormal(mean=1.0, stddev=0.5, seed=2021),
                                  trainable=True)
        if self.use_smooth_gate:
            # 50%旧信息 + 50%新信息
            self.smooth_gate_logit = self.add_weight(
                name='smooth_gate_logit',
                shape=(1,),
                initializer=keras.initializers.Constant(-0.5),
                trainable=True)
            # 平滑门控参数：控制「软更新比例」，平滑缩放因子，控制变化量的敏感度
            # smooth_scale > 1.0：放大变化，网络更"灵敏"
            # smooth_scale < 1.0：抑制变化，网络更"平滑"
            # smooth_scale ≈ 0.0：几乎忽略输入变化
            self.smooth_scale = self.add_weight(
                name='smooth_scale',
                shape=(1,),
                initializer=keras.initializers.Constant(0.5),
                trainable=True)

        super(Evolution, self).build(input_shape)  

    def call(self, all_data_static, threshold_nc, all_data_dynamic_now):
        # 1. 初始化动态特征序列：给初始动态特征加时间维度，形状从[N,dr2] → [1, N, dr2]
        all_data_dynamic = tf.expand_dims(all_data_dynamic_now, 0)
        # 2. 缓存初始动态特征，作为「旧特征」
        all_data_dynamic_now_prev = all_data_dynamic_now
        # 3. 计算【候选更新的动态特征】：核心的Update，decay过程，初始化
        all_data_dynamic_now_candidate = tf.sigmoid(
            tf.matmul(tf.concat([all_data_dynamic_now, all_data_static[0]], axis=-1), self.w1)
            * tf.repeat(threshold_nc[0], self.dr2, axis=-1) + all_data_dynamic_now
            * tf.repeat(1 - threshold_nc[0], self.dr2, axis=-1)) * math.exp(-1 / 2)
        if self.use_smooth_gate:
            delta0 = tf.reduce_mean(tf.abs(all_data_dynamic_now_candidate - all_data_dynamic_now_prev), axis=-1, keepdims=True)
            smooth_gate0 = tf.sigmoid(self.smooth_gate_logit + self.smooth_scale * delta0)
            all_data_dynamic_now = smooth_gate0 * all_data_dynamic_now_prev + (1 - smooth_gate0) * all_data_dynamic_now_candidate
        else:
            all_data_dynamic_now = all_data_dynamic_now_candidate
        all_data_dynamic_diff = []
        # 获取总时间步T
        time_steps = threshold_nc.shape[0]
        # 遍历从第1步到最后1步，逐时间步更新动态特征
        for i in range(1, time_steps):
            all_data_dynamic_now_diff = all_data_dynamic_now
            all_data_dynamic_now_prev = all_data_dynamic_now
             # 【和初始化阶段完全相同的核心更新逻辑】生成候选特征
            all_data_dynamic_now_candidate = tf.sigmoid(
                tf.matmul(tf.concat([all_data_dynamic_now, all_data_static[i]], axis=-1), self.w1)
                * tf.repeat(threshold_nc[i], self.dr2, axis=-1) + all_data_dynamic_now
                * tf.repeat(1 - threshold_nc[i], self.dr2, axis=-1)) * math.exp(-1 / 2)
            if self.use_smooth_gate:
                delta = tf.reduce_mean(tf.abs(all_data_dynamic_now_candidate - all_data_dynamic_now_prev), axis=-1, keepdims=True)
                smooth_gate = tf.sigmoid(self.smooth_gate_logit + self.smooth_scale * delta)
                all_data_dynamic_now = smooth_gate * all_data_dynamic_now_prev + (1 - smooth_gate) * all_data_dynamic_now_candidate
            else:
                all_data_dynamic_now = all_data_dynamic_now_candidate
            # 计算「特征差分」：当前特征 - 上一步特征 → 捕捉动态特征的变化率（流量变化/风险变化）
            all_data_dynamic_now_diff = all_data_dynamic_now - all_data_dynamic_now_diff
            all_data_dynamic_diff.append(tf.expand_dims(all_data_dynamic_now_diff, 0))
            # 把当前时间步的动态特征，拼接到完整序列中
            all_data_dynamic = tf.concat([all_data_dynamic, tf.expand_dims(all_data_dynamic_now, 0)], axis=0)
        # 最终输出
        if len(all_data_dynamic_diff) == 0:
            all_data_dynamic_diff = tf.zeros(
                (0, tf.shape(all_data_dynamic_now)[1], tf.shape(all_data_dynamic_now)[2]),
                dtype=all_data_dynamic_now.dtype,
            )
        else:
            all_data_dynamic_diff = tf.concat(all_data_dynamic_diff, axis=0)
        return all_data_dynamic, all_data_dynamic_now, all_data_dynamic_diff

# 空间表示学习部分
# 单分支的空间注意力机制，简单的说也就是让每个独立的交通网格，自适应学习周边邻居网格的特征
class Attention(Layer):

    def __init__(self, dr2, len_recent_time, number_region, mode='scaled_dot', **kwargs):
        # 统一的维度2D
        self.dr2 = dr2
        # 时间总步长，和动态层统一
        self.len_recent_time = len_recent_time
        # 城市网格数量
        self.number_region = number_region
        # 要学习的模式，有两种选择：scaled_dot, mean，如何把邻居网格的特征整合到当前网格
        self.mode = mode
        super(Attention, self).__init__(**kwargs)
    # 初始化【3 个核心可训练权重】
    def build(self, input_shape):
        self.wq = self.add_weight(
            shape=(self.dr2, self.dr2),
            initializer=keras.initializers.RandomNormal(mean=0.01, stddev=0.005, seed=2021),
            trainable=True)
        self.wk = self.add_weight(
            shape=(self.dr2, self.dr2),
            initializer=keras.initializers.RandomNormal(mean=0.01, stddev=0.005, seed=2021),
            trainable=True)
        self.wd_s = self.add_weight(
            shape=(self.dr2, self.dr2),
            initializer=keras.initializers.RandomNormal(mean=0.01, stddev=0.005, seed=2021),
            trainable=True)
        super(Attention, self).build(input_shape)  
    # 输入：data(特征张量), neigh_index(每个网格的邻居索引)
    def call(self, data, neigh_index):  
        # 核心操作：根据邻居索引，找到每个网格的所有邻居特征，实现「网格-邻居」的特征关联
        data_neigh = tf.nn.embedding_lookup(tf.transpose(data, (2, 0, 1, 3)),
                                            neigh_index)  
        # 张量形状变化：[N, len, T, dr2] → [N, K, len, T, dr2] → 每个网格都对应K个邻居的完整特征
        data_neigh = tf.transpose(data_neigh, (2, 3, 0, 1, 4)) 
        # 将自身特征重塑：增加「邻居维度」K=1，形状变为 [len, T, N, 1, dr2]，和邻居特征形状对齐
        data = tf.reshape(data, (-1, data.shape[1], data.shape[2], 1, data.shape[-1]))
        # 自身特征 → Query向量(q)：data * wq
        data = tf.matmul(data, self.wq)
        # 邻居特征 → Key向量(k)：data_neigh * wk （Value=Key，权重共享）
        data_neigh = tf.matmul(data_neigh, self.wk)
        # 低配版：均值池化，直接对邻居特征取平均，无自适应权重 → 等价于GCN的等权聚合
        if self.mode == 'mean':
            out = tf.reduce_mean(data_neigh, axis=-2, keepdims=True)
        else:
            logits = tf.matmul(data, data_neigh, transpose_b=True)
            # 核心版：缩放点积注意力（scaled_dot）→ 自适应学习邻居的「重要性权重」
            if self.mode == 'scaled_dot':
                logits = logits / tf.math.sqrt(tf.cast(self.dr2, logits.dtype))
            # softmax归一化：注意力分数→注意力权重，权重和为1，代表「每个邻居对当前网格的贡献度」
            out = tf.matmul(tf.nn.softmax(logits, axis=-1), data_neigh)
        # 残差连接
        out = data + out
        out = tf.sigmoid(
            tf.matmul(tf.reshape(out, (-1, self.len_recent_time, self.number_region, self.dr2)), self.wd_s))
        return out


class MultiAttention(Layer):

    def __init__(self, num_sp, dr2, len_recent_time, number_region, attention_mode='scaled_dot', **kwargs):
        self.dr2 = dr2
        self.num_sp = num_sp
        self.attention_layers_poi = [Attention(self.dr2, len_recent_time, number_region, mode=attention_mode) for _ in range(self.num_sp)]
        self.attention_layers_road = [Attention(self.dr2, len_recent_time, number_region, mode=attention_mode) for _ in range(self.num_sp)]
        self.attention_layers_record = [Attention(self.dr2, len_recent_time, number_region, mode=attention_mode) for _ in range(self.num_sp)]

        super(MultiAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_poi = self.add_weight(
            shape=(self.dr2,),
            initializer=keras.initializers.Zeros(),
            trainable=True)
        self.w_road = self.add_weight(
            shape=(self.dr2,),
            initializer=keras.initializers.Zeros(),
            trainable=True)
        self.w_record = self.add_weight(
            shape=(self.dr2,),
            initializer=keras.initializers.Zeros(),
            trainable=True)
        super(MultiAttention, self).build(input_shape)

    def call(self, all_data, neigh_poi_index, neigh_road_index, neigh_record_index):
        # 初始化三个分支的输入特征，都是同一个融合特征
        all_data_static_poi = all_data
        all_data_static_road = all_data
        all_data_static_record = all_data
        # 逐轮执行注意力学习：每类特征做num_sp轮空间学习，特征越来越精准
        for i in range(self.num_sp):
            all_data_static_poi = self.attention_layers_poi[i](all_data_static_poi, neigh_poi_index)
            all_data_static_road = self.attention_layers_road[i](all_data_static_road, neigh_road_index)
            all_data_static_record = self.attention_layers_record[i](all_data_static_record, neigh_record_index)
        # 核心：三类分支特征 加权相加 + Sigmoid激活 → 最终的空间融合特征
        out = tf.sigmoid(all_data_static_poi * self.w_poi + all_data_static_road * self.w_road +
                         all_data_static_record * self.w_record)
        return out


class MYPLAN(tf.keras.models.Model):
    def __init__(self, dr, len_recent_time, number_sp, number_region, neigh_poi_index, neigh_road_index,
                 neigh_record_index, attention_mode='scaled_dot', evolution_smooth=True, **kwargs):
        super(MYPLAN, self).__init__(**kwargs)
        # ========== 第一步：保存「固定不变的参数/索引」，全程复用 ==========
        self.neigh_poi_index = neigh_poi_index
        self.neigh_road_index = neigh_road_index
        self.neigh_record_index = neigh_record_index
        # ========== 第二步：实例化【动态特征演化核心层】 ==========
        self.evolution = Evolution(dr * 2, use_smooth_gate=bool(evolution_smooth))
        # ========== 第三步：实例化【空间注意力层×2】 ==========
        self.multiattention = [
            MultiAttention(number_sp, 2 * dr, len_recent_time, number_region, attention_mode=attention_mode)
            for _ in range(2)
        ]
        # 核心设计：创建2个完全独立的MultiAttention空间注意力层！
        # 分工明确：第0个 → 专门处理【动态特征】的空间学习；第1个 → 专门处理【静态特征】的空间学习
        # 为什么分离？静态特征(POI/道路)和动态特征(流量/风险)的「空间依赖规律完全不同」，分开学习更精准，不会互相干扰

        # ========== 第四步：实例化【模型核心 - ConvLSTM2D 时空建模层】 ==========，后续还会和普通LSTM作比较
        self.convlstm = keras.layers.ConvLSTM2D(1, 1, strides=(1, 1),
                                                padding='valid',
                                                data_format=None,
                                                dilation_rate=(1, 1),
                                                activation='tanh',
                                                recurrent_activation='hard_sigmoid',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                recurrent_initializer='orthogonal',
                                                bias_initializer='zeros',
                                                unit_forget_bias=True,
                                                return_sequences=False,
                                                )
        # ========== 第五步：实例化【最终输出层 - Dense全连接层】 ==========
        self.final_layer = keras.layers.Dense(number_region, activation='sigmoid', bias_initializer='ones')
    # 调用前面设计的所有环节
    def call(self, all_data_static, threshold_nc, all_data_dynamic_now):
        all_data_dynamic, all_data_dynamic_now, all_data_dynamic_diff = self.evolution(all_data_static, threshold_nc,
                                                                                       all_data_dynamic_now)

        all_data_dynamic = self.multiattention[0](all_data_dynamic, self.neigh_poi_index, self.neigh_road_index,
                                                  self.neigh_record_index)
        all_data_static = self.multiattention[1](all_data_static, self.neigh_poi_index, self.neigh_road_index,
                                                 self.neigh_record_index)
        all_data_dynamic = tf.expand_dims(all_data_dynamic, 3)
        all_data_static = tf.expand_dims(all_data_static, 3)
        all_data = tf.concat([all_data_dynamic, all_data_static], axis=-1)
        all_data = self.convlstm(all_data)
        all_data = tf.reshape(all_data, (-1, all_data.shape[1]))
        all_data = self.final_layer(all_data)
        # print(all_data.shape)
        return all_data, all_data_dynamic_now, all_data_dynamic_diff


class BaselineRNN(tf.keras.models.Model):

    def __init__(self, dr, len_recent_time, number_region, rnn_type='lstm', **kwargs):
        super(BaselineRNN, self).__init__(**kwargs)
        self.dr = dr
        self.len_recent_time = len_recent_time
        self.number_region = number_region
        self.rnn_type = str(rnn_type).lower()

        units = 2 * dr
        if self.rnn_type == 'gru':
            self.rnn = keras.layers.GRU(units, return_sequences=False)
        else:
            self.rnn = keras.layers.LSTM(units, return_sequences=False)

        self.out = keras.layers.Dense(1, activation='sigmoid')

    def call(self, all_data_static, threshold_nc, all_data_dynamic_now):
        # all_data_static: [B, T, R, F]
        x = all_data_static
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        r = tf.shape(x)[2]
        f = tf.shape(x)[3]
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (b * r, t, f))
        h = self.rnn(x)
        y = self.out(h)
        y = tf.reshape(y, (b, r))

        # Keep signature compatible with MYPLAN
        dy_diff = tf.zeros((t - 1, r, 2 * self.dr), dtype=y.dtype)
        return y, all_data_dynamic_now, dy_diff


class BaselineMLP(tf.keras.models.Model):

    def __init__(self, dr, len_recent_time, number_region, **kwargs):
        super(BaselineMLP, self).__init__(**kwargs)
        self.dr = dr
        self.len_recent_time = len_recent_time
        self.number_region = number_region

        hidden = 4 * dr
        self.fc1 = keras.layers.Dense(hidden, activation='relu')
        self.fc2 = keras.layers.Dense(hidden, activation='relu')
        self.out = keras.layers.Dense(1, activation='sigmoid')

    def call(self, all_data_static, threshold_nc, all_data_dynamic_now):
        # all_data_static: [B, T, R, F]
        x = all_data_static
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        r = tf.shape(x)[2]
        f = tf.shape(x)[3]
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (b * r, t * f))
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.out(x)
        y = tf.reshape(y, (b, r))

        dy_diff = tf.zeros((t - 1, r, 2 * self.dr), dtype=y.dtype)
        return y, all_data_dynamic_now, dy_diff
