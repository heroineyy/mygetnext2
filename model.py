import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class GenericEmbeddings(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(GenericEmbeddings, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
        )

    def forward(self, idx):
        embed = self.embedding(idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x

class FuseEmbeddings3(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim, fusion_type='concat'):
        """
        改进的嵌入融合模块，支持多种融合方式

        参数:
            user_embed_dim: 用户嵌入维度
            poi_embed_dim: POI嵌入维度
            fusion_type: 融合方式，可选 'concat', 'sum', 'weighted_sum', 'product', 'concat_weighted_sum'
        """
        super(FuseEmbeddings3, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            embed_dim = user_embed_dim + poi_embed_dim
            self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        elif fusion_type == 'sum':
            assert user_embed_dim == poi_embed_dim, "For sum fusion, dimensions must match"
            embed_dim = user_embed_dim
            self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        elif fusion_type == 'weighted_sum':
            assert user_embed_dim == poi_embed_dim, "For weighted sum, dimensions must match"
            embed_dim = user_embed_dim
            self.user_weight = nn.Parameter(torch.rand(1))
            self.poi_weight = nn.Parameter(torch.rand(1))
            self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        elif fusion_type == 'product':
            assert user_embed_dim == poi_embed_dim, "For product fusion, dimensions must match"
            embed_dim = user_embed_dim
            self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        elif fusion_type == 'concat_weighted_sum':
            embed_dim = user_embed_dim + poi_embed_dim
            # 先拼接
            self.concat_fuse = nn.Linear(embed_dim, embed_dim)
            # 再加权相加（如果维度相同）
            if user_embed_dim == poi_embed_dim:
                self.user_weight = nn.Parameter(torch.rand(1))
                self.poi_weight = nn.Parameter(torch.rand(1))
                self.final_fuse = nn.Linear(embed_dim+user_embed_dim, embed_dim)
            else:
                self.final_fuse = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        if self.fusion_type == 'concat':
            x = torch.cat((user_embed, poi_embed), dim=-1)
            x = self.fuse_embed(x)
        elif self.fusion_type == 'sum':
            x = user_embed + poi_embed
            x = self.fuse_embed(x)
        elif self.fusion_type == 'weighted_sum':
            # 使用softmax确保权重和为1
            weights = torch.softmax(torch.stack([self.user_weight, self.poi_weight]), dim=0)
            x = weights[0] * user_embed + weights[1] * poi_embed
            x = self.fuse_embed(x)
        elif self.fusion_type == 'product':
            x = user_embed * poi_embed
            x = self.fuse_embed(x)
        elif self.fusion_type == 'concat_weighted_sum':
            # 拼接部分
            concat_x = torch.cat((user_embed, poi_embed), dim=-1)
            concat_x = self.concat_fuse(concat_x)

            # 加权相加部分（如果维度相同）
            if user_embed.shape[-1] == poi_embed.shape[-1]:
                weights = torch.softmax(torch.stack([self.user_weight, self.poi_weight]), dim=0)
                weighted_x = weights[0] * user_embed + weights[1] * poi_embed
                # 将加权相加结果与拼接结果相加
                # x = concat_x + weighted_x
                x = torch.cat((concat_x, weighted_x), dim=-1)
            else:
                x = concat_x

            x = self.final_fuse(x)

        x = self.leaky_relu(x)
        return x

class FuseEmbeddings4(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3):
        """
        改进的嵌入融合模块，加权融合,其中forward会传入(self, embeding1, embeding2,embeding3,embeding4)
        其中embeding1和embeding2的维度相同会加权相加,embeding3,embeding4的维度相同也会加权相加

        最后分别相加的两个向量会拼接再使用concat_fuse得到最后的向量
        """
        super(FuseEmbeddings4, self).__init__()
        self.weight1 = nn.Parameter(torch.rand(1))
        self.weight2 = nn.Parameter(torch.rand(1))
        self.weight3 = nn.Parameter(torch.rand(1))
        self.weight4 = nn.Parameter(torch.rand(1))
        # 确保输入维度设置正确，这里计算拼接后的维度
        embed_dim = embed_dim1 + embed_dim2 + embed_dim3
        self.concat_fuse = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embeding0, embeding1, embeding2, embeding3, embeding4):
        # 对 embeding1 和 embeding2 进行加权相加
        weighted_sum_1_2 = self.weight1 * embeding1 + self.weight2 * embeding2
        # 对 embeding3 和 embeding4 进行加权相加
        weighted_sum_3_4 = self.weight3 * embeding3 + self.weight4 * embeding4

        # 将两个加权和向量进行拼接
        concat_embedding = torch.cat((embeding0, weighted_sum_1_2, weighted_sum_3_4), dim=-1)

        # 通过线性层进行融合
        fused_embedding = self.concat_fuse(concat_embedding)

        # 通过激活函数
        output = self.leaky_relu(fused_embedding)


        return output


class FuseEmbeddings5(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3):
        """
        改进的嵌入融合模块，加权融合,其中forward会传入(self, embeding1, embeding2,embeding3,embeding4)
        其中embeding1和embeding2的维度相同会加权相加,embeding3,embeding4的维度相同也会加权相加

        最后分别相加的两个向量会拼接再使用concat_fuse得到最后的向量
        """
        super(FuseEmbeddings5, self).__init__()
        self.weight1 = nn.Parameter(torch.rand(1))
        self.weight2 = nn.Parameter(torch.rand(1))
        # 确保输入维度设置正确，这里计算拼接后的维度
        embed_dim = embed_dim1 + embed_dim2 + embed_dim3
        self.concat_fuse = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embeding0, embeding1, embeding2):

        # 将两个加权和向量进行拼接
        concat_embedding = torch.cat((embeding0, self.weight1 * embeding1, self.weight2 * embeding2), dim=-1)

        # 通过线性层进行融合
        fused_embedding = self.concat_fuse(concat_embedding)

        # 通过激活函数
        output = self.leaky_relu(fused_embedding)

        return output
class FuseEmbeddings2(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3):
        super(FuseEmbeddings2, self).__init__()
        embed_dim = embed_dim1 + embed_dim2 + embed_dim3
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embeding1, embeding2,embeding3):
        x = self.fuse_embed(torch.cat((embeding1, embeding2,embeding3), 0))
        x = self.leaky_relu(x)
        return x

class AttentionFuseEmbeddings(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, embed_dim3):
        super(AttentionFuseEmbeddings, self).__init__()
        self.embed_dim = embed_dim1 + embed_dim2 + embed_dim3
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.fuse_embed = nn.Linear(self.embed_dim, self.embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, embeding1, embeding2, embeding3):
        x = torch.cat((embeding1, embeding2, embeding3), dim=-1)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        weighted_output = torch.matmul(attention_probs, v)
        x = self.fuse_embed(weighted_output)
        x = self.leaky_relu(x)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query)  # [embed_size]
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)  # [len, 1]
        weight = torch.unsqueeze(weight, 1)
        temp2 = torch.mul(value, weight)
        out = torch.sum(temp2, 0)  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out
        
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float32))
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, V)
        return output


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_lat = nn.Linear(embed_size, 1)
        self.decoder_lon = nn.Linear(embed_size, 1)
        self.decoder_week = nn.Linear(embed_size, 7)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.decoder_categroy = nn.Linear(embed_size, 11)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_lat = self.decoder_lat(x)
        out_lon = self.decoder_lon(x)
        out_week = self.decoder_week(x)
        out_cat = self.decoder_cat(x)
        out_categroy = self.decoder_categroy(x)
        return out_poi, out_time, out_lat,out_lon,out_cat,out_categroy,out_week

# 定义一个模拟大语言模型预测的函数
# def llm_prediction(input_categories):
#     # 这里只是简单模拟大语言模型的输出，实际应用中需要调用真实的大语言模型
#     # 构建提示词模板，将用户轨迹转换成自然语言
#     prompt = f"用户之前访问的类别序列是：{', '.join(input_categories)}，预测下一个访问点的类别。"
#     # 这里简单返回一个模拟的预测结果
#     prediction = "E"
#     return prediction

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 添加POI特征融合模块
class POIFeatureFusion(nn.Module):
    def __init__(self, poi_embed_dim, fuse_way='adaptive'):
        super(POIFeatureFusion, self).__init__()
        self.fuse_way = fuse_way
        self.poi_embed_dim = poi_embed_dim
        
        if fuse_way == 'adaptive':
            # 自适应权重学习
            self.weight_net = nn.Sequential(
                nn.Linear(poi_embed_dim * 2, poi_embed_dim),
                nn.ReLU(),
                nn.Linear(poi_embed_dim, 2),
                nn.Softmax(dim=-1)
            )
        elif fuse_way == 'concat':
            # 拼接后对齐
            self.align_layer = nn.Linear(poi_embed_dim * 2, poi_embed_dim)
        elif fuse_way == 'manual':
            # 手动加权
            self.weight_poi = nn.Parameter(torch.tensor(0.5))
            self.weight_gcn = nn.Parameter(torch.tensor(0.5))
        else:
            raise ValueError(f"Unsupported fusion way: {fuse_way}")
        
    def forward(self, poi_embedding, poi_gcn_embedding):
        fused_embedding = poi_embedding
        if self.fuse_way == 'adaptive':
            # 自适应权重
            concat_features = torch.cat([poi_embedding, poi_gcn_embedding], dim=-1)
            weights = self.weight_net(concat_features)
            fused_embedding = weights[0] * poi_embedding + weights[1] * poi_gcn_embedding
            
        elif self.fuse_way == 'concat':
            # 拼接后对齐
            concat_features = torch.cat([poi_embedding, poi_gcn_embedding], dim=-1)
            fused_embedding = self.align_layer(concat_features)
            
        elif self.fuse_way == 'manual':
            # 手动加权
            weights = torch.sigmoid(torch.stack([self.weight_poi, self.weight_gcn]))
            fused_embedding = weights[0] * poi_embedding + weights[1] * poi_gcn_embedding
            
        return fused_embedding
    
class EnhancedUserEmbedding(nn.Module):
    def __init__(self, num_users, embed_dim, cooccurrence_matrix, k=5):
        super(EnhancedUserEmbedding, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.cooccurrence_matrix = cooccurrence_matrix
        self.k = k  # 每个用户考虑的相似用户数量
        
    def get_similar_users(self, user_idx):
        """获取与目标用户最相似的k个用户"""
        similar_users = torch.topk(self.cooccurrence_matrix[user_idx], self.k + 1)[1][1:]  # 去掉自己
        return similar_users
    
    def forward(self, user_idx):
        
        base_embedding = self.user_embedding(user_idx)
            
        similar_users = self.get_similar_users(user_idx)
        similar_embeddings = self.user_embedding(similar_users)
            
        # 3. 计算相似用户的加权平均
        similarities = self.cooccurrence_matrix[user_idx][similar_users]
        weights = F.softmax(similarities, dim=0)
        similar_embedding = torch.sum(weights.unsqueeze(1) * similar_embeddings, dim=0)
            
        # 4. 融合基础嵌入和相似用户嵌入
        enhanced_embedding = 0.7 * base_embedding + 0.3 * similar_embedding
            
        return enhanced_embedding

# class LLMFeatureExtractor:
#     def __init__(self, model_type="llama3", device="cuda"):
#         self.model_type = model_type
#         self.device = device
#         if model_type == "nomic-embed-text":
#             self.model_url = "http://localhost:11434/api/embeddings"
#         else:
#             self.model_url = "http://localhost:11434/api/generate"
#
#     def get_embedding(self, prompt, poi_embed_model=None):
#         try:
#             if self.model_type == "nomic-embed-text":
#                 response = requests.post(
#                     self.model_url,
#                     json={
#                         "model": "nomic-embed-text",
#                         "prompt": prompt
#                     }
#                 )
#                 if response.status_code == 200:
#                     return response.json()["embedding"]
#                 else:
#                     print(f"Error getting embedding: {response.status_code}")
#                     return None
#             # TODO:大模型输出不一定是id,而是一段文字所以可能会出现问题,还有输入部分时间是被处理的,他是一个数字,
#             # 比如0.8表示的是19:00,所以需要处理,另外day其实是星期数,所以需要处理,给大模型强调一定要输出类别,所以需要给大模型建立知识库,这里打算用dify,启动dify,建立知识库限定大模型只能从这个知识库里面的类别里面找
#             elif self.model_type == "llama3":
#                 response = requests.post(
#                     self.model_url,
#                     json={
#                         "model": "llama3.1",
#                         "prompt": prompt,
#                         "stream": False
#                     }
#                 )
#                 if response.status_code == 200:
#                     response_data = response.json()
#                     predicted_category = response_data.get("response", "").strip()
#                     poi_id = self._category_to_poi_id(predicted_category)
#                     if poi_embed_model is not None:
#                         return poi_embed_model.get_embedding(poi_id)
#                     return None
#                 else:
#                     print(f"Error getting response: {response.status_code}")
#                     print(f"Response content: {response.text}")
#                     return None
#         except Exception as e:
#             print(f"Error in get_embedding: {str(e)}")
#             return None
#
#     def _category_to_poi_id(self, category):
#         """Convert category name to POI ID using existing dictionaries"""
#         try:
#             # 首先尝试在cat_name2id_dict中查找
#             if category in cat_name2id_dict:
#                 return cat_name2id_dict[category]
#
#             # 如果找不到，尝试在category_name2id_dict中查找
#             if category in category_name2id_dict:
#                 return category_name2id_dict[category]
#
#             # 如果都找不到，打印警告并返回默认值
#             print(f"Warning: Category '{category}' not found in dictionaries")
#             return 0
#
#         except Exception as e:
#             print(f"Error in _category_to_poi_id: {e}")
#             return 0

class GatedExpertNetwork(nn.Module):
    def __init__(self, embed_size, num_experts=2):
        super(GatedExpertNetwork, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(embed_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x shape: [seq_len, batch_size, embed_size]
        x = x.transpose(0, 1)
        gate_weights = self.gate(x)  # [seq_len, batch_size, num_experts]
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [seq_len, batch_size, embed_size, num_experts]
        output = torch.einsum('sben,sbn->sbe', expert_outputs, gate_weights)
        output = output.transpose(0, 1)
        return output

class EnhancedTransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(EnhancedTransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # 保持原有的初始化代码
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embed_size = embed_size
        self.gated_expert = GatedExpertNetwork(embed_size)
        self.self_attention = nn.MultiheadAttention(embed_size, nhead)
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_lat = nn.Linear(embed_size, 1)
        self.decoder_lon = nn.Linear(embed_size, 1)
        self.decoder_week = nn.Linear(embed_size, 7)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.decoder_categroy = nn.Linear(embed_size, 11)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        x = self.gated_expert(x)
        x, _ = self.self_attention(x, x, x)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_lat = self.decoder_lat(x)
        out_lon = self.decoder_lon(x)
        out_week = self.decoder_week(x)
        out_cat = self.decoder_cat(x)
        out_categroy = self.decoder_categroy(x)
        return out_poi, out_time, out_lat, out_lon, out_cat, out_categroy, out_week
