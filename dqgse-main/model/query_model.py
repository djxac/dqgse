import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaModel


class TextImageFusionModel(nn.Module):
    def __init__(self,device, bert_model_name,num_layers, num_queries, query_dim, embed_dim, num_heads, hidden_dim,entity_type_count,dropout=0.1):
        super(TextImageFusionModel, self).__init__()
        # 加载预训练的 BERT 模型
        self.tokenizer = RobertaTokenizer.from_pretrained('../lqmasa/roberta-base-cased')

        self.bert = RobertaModel.from_pretrained('../lqmasa/roberta-base-cased')
        # 定义可学习的查询向量
        self.queries = nn.Embedding(num_queries,query_dim*2)

        sentences = ["Person is an entity type about [MASK]",
                     "Organization is an entity type about [MASK]",
                     "Location is an entity type about [MASK]",
                     ] * (num_queries//3)
        sentences_id=self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert(**sentences_id)
        self.sentence_representations = torch.mean(outputs.last_hidden_state[:,1:-1,:].detach(),dim=1).to(device)
        self.layers = nn.ModuleList(
             [TextImageFusionLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])

        #self.img_null_embedding = nn.Parameter(torch.randn(1, 768))

    # def add_null_token(self, features, null_embedding):
    #     """
    #     向特征序列添加空标记
    #     Args:
    #         features: 输入特征，形状为 [batch_size, seq_len, feat_dim]
    #         null_embedding: 空标记嵌入，形状为 [1, feat_dim]
    #     Returns:
    #         添加空标记后的特征，形状为 [batch_size, seq_len+1, feat_dim]
    #     """
    #     batch_size = features.size(0)
    #     # 扩展空标记到批次大小
    #     null_tokens = null_embedding.repeat(batch_size, 1, 1)  # [batch_size, 1, feat_dim]
    #     # 在序列开头添加空标记
    #     return torch.cat([null_tokens, features], dim=1)

    def forward(self, text_hidden_states, text_attention_mask, vision_hidden_states, vision_attention_mask,cap_input_ids,cap_attention_mask):
        queries = self.queries.weight
        pos, tgt = torch.split(queries, text_hidden_states.size(2), dim=-1)
        tgt=tgt+self.sentence_representations
        #tgt=self.sentence_representations
        tgt = tgt.unsqueeze(0).expand(text_hidden_states.size(0), -1, -1)
        pos = pos.unsqueeze(0).expand(text_hidden_states.size(0), -1, -1)

        #vision_hidden_states=self.add_null_token(vision_hidden_states, self.img_null_embedding)
        #cap_hidden_states = self.bert(cap_input_ids, attention_mask=cap_attention_mask)["last_hidden_state"]

        for layer in self.layers:
            output_queries, text_queries, image_queries= layer(tgt,pos, text_hidden_states,
                                                        vision_hidden_states,text_attention_mask,vision_attention_mask,None,None)
            tgt=output_queries

        return tgt,text_queries,image_queries

class TextSentimentLayer(nn.Module):
    def __init__(self,embed_dim, num_heads, hidden_dim):
        super(TextSentimentLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads=num_heads)
        self.sublayer_connect0=SublayerConnection(size=embed_dim)
        self.text_cross_attention = CrossAttention(embed_dim, num_heads=num_heads)
        self.sublayer_connect1 = SublayerConnection(size=embed_dim)
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    def forward(self, tgt, pos, text_hidden_states,text_attention_mask):
        batch_size = text_hidden_states.size(0)

        # 创建 Transformer 编码器的掩码
        query_mask = torch.ones(batch_size, tgt.size(1), dtype=text_attention_mask.dtype, device=text_attention_mask.device)
        # 通过自注意力层融合查询向量和文本隐层状态
        query_self=self.self_attention(query=self.with_pos_embed(tgt,pos),key=self.with_pos_embed(tgt,pos),value=tgt,attention_mask=query_mask)
        query_self = self.sublayer_connect0(x=tgt, sublayer_x=query_self)

        text_cross_output_only=self.text_cross_attention(query=self.with_pos_embed(query_self,pos),key=text_hidden_states,value=text_hidden_states,attention_mask=text_attention_mask)
        #text_cross_output=self.sublayer_connect1(x=tgt,sublayer_x=text_cross_output_only)
        return text_cross_output_only

class TextImageFusionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads,hidden_dim):
        super(TextImageFusionLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads=num_heads)
        self.sublayer_connect0=SublayerConnection(size=embed_dim)
        self.text_cross_attention=CrossAttention(embed_dim, num_heads=num_heads)
        self.sublayer_connect1 = SublayerConnection(size=embed_dim)
        # 定义交叉注意力层
        self.image_cross_attention = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.sublayer_connect2 = SublayerConnection(size=embed_dim)

        #self.cap_cross_attention = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        # 定义前馈神经网络
        # self.feed_forward = FeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim)
        # self.sublayer_connect3 = SublayerConnection(size=embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.k = 48
        #self.dynamic_gating = UnifiedWeightGating(embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def compute_global_features(self, text_hidden_states, vision_hidden_states, text_attention_mask,
                                vision_attention_mask):
        """
        使用 attention mask 对有效 token 做 masked mean pooling 得到全局特征
        """
        # 文本全局特征
        if text_attention_mask is not None:
            # 扩展 mask 以匹配特征维度 [B, T, 1]
            mask_expanded = text_attention_mask.unsqueeze(-1).float()
            text_sum = (text_hidden_states * mask_expanded).sum(dim=1)
            text_count = mask_expanded.sum(dim=1)
            text_global = text_sum / (text_count + 1e-8)  # [B, D]
        else:
            text_global = text_hidden_states.mean(dim=1)  # [B, D]

        # 图像全局特征
        if vision_attention_mask is not None:
            mask_expanded = vision_attention_mask.unsqueeze(-1).float()
            img_sum = (vision_hidden_states * mask_expanded).sum(dim=1)
            img_count = mask_expanded.sum(dim=1)
            img_global = img_sum / (img_count + 1e-8)
        else:
            img_global = vision_hidden_states.mean(dim=1)

        return text_global, img_global  # [B, D], [B, D]

    def differentiable_topk(self, scores, k, temperature=0.1):
        """
        使用 Gumbel-Softmax + Straight-Through 实现可微 top-k
        scores: [B, N] 每个查询的得分
        return: hard_mask [B, N], indices [B, k]
        """
        u = torch.rand_like(scores)
        gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        scores_gumbel = scores + gumbel_noise  # [B, N]

        # Softmax 得到 soft 概率
        soft_mask = F.softmax(scores_gumbel / temperature, dim=-1)  # [B, N]

        # 获取真实 top-k 索引（用于后续选择）
        _, indices = torch.topk(scores, k, dim=-1, sorted=False)  # [B, k]

        # 构造 one-hot 硬掩码
        hard_mask = torch.zeros_like(scores).scatter_(-1, indices, 1.0)  # [B, N]

        # Straight-Through: 前向用 hard，反向用 soft
        hard_mask = (hard_mask - soft_mask).detach() + soft_mask  # [B, N]

        return hard_mask, indices

    def forward(self, tgt,pos, text_hidden_states, vision_hidden_states,text_attention_mask,vision_attention_mask,cap_hidden_states,cap_attention_mask):
        batch_size = text_hidden_states.size(0)

        # 创建 Transformer 编码器的掩码
        query_mask = torch.ones(batch_size, tgt.size(1), dtype=text_attention_mask.dtype, device=text_attention_mask.device)
        # 通过自注意力层融合查询向量和文本隐层状态
        query_self=self.self_attention(query=self.with_pos_embed(tgt,pos),key=self.with_pos_embed(tgt,pos),value=tgt,attention_mask=query_mask)
        query_self = self.sublayer_connect0(x=tgt, sublayer_x=query_self)

        text_cross_output_only=self.text_cross_attention(query=self.with_pos_embed(query_self,pos),key=text_hidden_states,value=text_hidden_states,attention_mask=text_attention_mask)
        text_cross_output=self.sublayer_connect1(x=query_self,sublayer_x=text_cross_output_only)
        #通过交叉注意力层融合查询向量和视觉隐层状态
        image_cross_output_only= self.image_cross_attention(query=self.with_pos_embed(text_cross_output,pos), key=vision_hidden_states,
                                                 value=vision_hidden_states, attention_mask=vision_attention_mask)
        image_cross_output = self.sublayer_connect2(x=query_self, sublayer_x=text_cross_output_only+image_cross_output_only)
        # 通过前馈神经网络进一步融合
        # fused_output = self.feed_forward(image_cross_output)
        # fused_output = self.sublayer_connect3(x=image_cross_output,sublayer_x=fused_output)
        return image_cross_output,text_cross_output_only,image_cross_output_only

    def calculate_consistency_score(self, text_global_feat, visual_global_feat):
        """
        计算一致性分数a_v：完全遵循文档1-47节的余弦相似度公式
        公式：a_v = (f_t · e_v) / (||f_t|| × ||e_v||)

        Args:
            text_global_feat: 聚合后的文本全局特征，形状[1, 768]（对应文档f_t）
            visual_global_feat: 聚合后的视觉全局特征，形状[1, 768]（对应文档e_v）
        Returns:
            a_v: 一致性分数，标量（取值范围[-1,1]，越接近1表示特征一致性越高）
        """
        # 计算点积：f_t · e_v
        dot_product = torch.matmul(text_global_feat, visual_global_feat.transpose(1,2))  # 结果形状[1,1]

        # 计算L2范数：||f_t||和||e_v||
        text_norm = torch.norm(text_global_feat, p=2, dim=2, keepdim=True)  # 形状[1,1]
        visual_norm = torch.norm(visual_global_feat, p=2, dim=2, keepdim=True)  # 形状[1,1]

        # 计算一致性分数
        a_v = dot_product / (text_norm * visual_norm)
        return a_v  # 压缩为标量

class UnifiedWeightGating(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(UnifiedWeightGating, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.text_norm = nn.LayerNorm(input_dim)
        self.image_norm = nn.LayerNorm(input_dim)

        # 用于计算全局相似度的池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 多头门控参数
        self.W_a_heads = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_heads)])
        self.W_q_heads = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_heads)])

        # 处理全局相似度的线性层
        self.similarity_proj = nn.Linear(1, self.head_dim)

        self.activation = nn.ReLU()
        self.fc = nn.Linear(num_heads * self.head_dim, 1)  # 输出单个权重值

    def forward(self, text_representation, image_representation):
        # 输入形状: (batch_size, 96, dim)
        batch_size, seq_len, dim = text_representation.shape

        # 标准化操作
        text_rep = self.text_norm(text_representation)  # (batch_size, 96, dim)
        image_rep = self.image_norm(image_representation)  # (batch_size, 96, dim)

        # 计算全局相似度
        per_pos_sim = F.cosine_similarity(text_rep, image_rep, dim=-1, eps=1e-6)  # (batch_size, 96)
        global_sim = self.global_pool(per_pos_sim.unsqueeze(1)).squeeze(1).unsqueeze(-1)  # (batch_size, 1)

        # 计算全局门控
        text_global = self.global_pool(text_rep.transpose(1, 2)).transpose(1, 2).squeeze(1)  # (batch_size, dim)
        image_global = self.global_pool(image_rep.transpose(1, 2)).transpose(1, 2).squeeze(1)  # (batch_size, dim)

        # 多头门控计算
        head_gates = []
        for i in range(self.num_heads):
            gate_a = self.W_a_heads[i](text_global)  # (batch_size, head_dim)
            gate_q = self.W_q_heads[i](image_global)  # (batch_size, head_dim)
            sim_feat = self.similarity_proj(global_sim)# (batch_size, head_dim)
            sim_feat = sim_feat.squeeze(1)
            gate = self.activation(gate_a + gate_q + sim_feat)  # (batch_size, head_dim)
            head_gates.append(gate)

        # 拼接多头结果并计算最终门控值
        combined_gates = torch.cat(head_gates, dim=-1)  # (batch_size, num_heads*head_dim)
        final_gate = self.fc(combined_gates)  # (batch_size, 1)
        final_gate = torch.sigmoid(final_gate)  # 确保在0-1范围内

        # 修复：使用更稳健的方式扩展维度，避免硬编码形状
        # 从final_gate中获取实际的batch_size，而不是依赖输入的batch_size
        actual_batch_size = final_gate.size(0)

        # 重塑为三维张量 (actual_batch_size, 1, 1)
        final_gate = final_gate.view(actual_batch_size, 1, 1)

        # 扩展到与输入序列匹配的形状
        final_gate = final_gate.expand(-1, seq_len, dim)  # 使用expand更高效，自动处理维度

        # 融合两个模态
        fused_representation = final_gate * text_rep + (1 - final_gate) * image_rep

        return fused_representation


class MultiModalDynamicGating(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiModalDynamicGating, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        # 每个头的维度
        self.head_dim = input_dim // num_heads

        self.text_norm = nn.LayerNorm(768)
        self.image_norm = nn.LayerNorm(768)

        # 每个头的权重矩阵
        self.W_a_heads = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_heads)])
        self.W_q_heads = nn.ModuleList([nn.Linear(input_dim, self.head_dim) for _ in range(num_heads)])

        # 非线性层
        self.activation = nn.ReLU()

        # 最终的线性层用于融合多头结果
        self.fc = nn.Linear(num_heads * self.head_dim, input_dim)

    def forward(self, text_representation, image_representation):
        # 存储每个头的门控结果
        head_gates = []
        text_representation = self.text_norm(text_representation)
        image_representation = self.image_norm(image_representation)
        for i in range(self.num_heads):
            # 计算每个头的门控
            gate_a = self.W_a_heads[i](text_representation)
            gate_q = self.W_q_heads[i](image_representation)

            # 非线性变换
            gate = self.activation(gate_a + gate_q)

            # 逐元素sigmoid操作
            gate = torch.sigmoid(gate)

            head_gates.append(gate)

        # 拼接所有头的结果
        combined_gates = torch.cat(head_gates, dim=-1)

        # 融合多头结果
        final_gate = self.fc(combined_gates)

        # 动态分配权重
        fused_representation = final_gate * text_representation + (1 - final_gate) * image_representation

        return fused_representation
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention =nn.MultiheadAttention(embed_dim, num_heads,dropout=0.1)

    def forward(self, query, key, value, attention_mask=None):
        # query, key, value: (batch_size, seq_len, embed_dim)
        # attention_mask: (batch_size, seq_len)
        # 计算自注意力
        attn_output,_ = self.attention(query.transpose(0,1), key.transpose(0,1), value.transpose(0,1),key_padding_mask=~(attention_mask.bool()))

        return attn_output.transpose(0,1)
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim,dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention =nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,dropout=0.1)

    def forward(self, query, key, value, attention_mask=None):
        # query: (batch_size, query_seq_len, embed_dim)
        # key, value: (batch_size, key_seq_len, embed_dim)
        # attention_mask: (batch_size, key_seq_len)
        # 计算交叉注意力
        attn_output,_= self.attention(query.transpose(0,1), key.transpose(0,1), value.transpose(0,1), key_padding_mask=~(attention_mask.bool()))
        # attn_output, attn_weights = self.attention(query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
        #                                 key_padding_mask=~(attention_mask.bool()),
        #                                 need_weights=True)
        return attn_output.transpose(0,1)
class SublayerConnection(nn.Module):
    """
    残差连接和层归一化（Add&Norm）模块。
    """
    def __init__(self, size, dropout=0.1):
        """
        :param size: 输入特征的维度。
        :param dropout: Dropout 概率。
        """
        super(SublayerConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(size)  # 层归一化模块
        self.dropout = nn.Dropout(p=dropout)  # Dropout 模块

    def forward(self, x, sublayer_x):
        """
        前向传播。
        :param x: 输入张量，形状为 (batch_size, seq_len, feature)。
        :param sublayer: 子层模块，例如多头注意力模块。
        :return: 残差连接和归一化后的张量。
        """
        return self.layer_norm(x + self.dropout(sublayer_x))  # 残差连接和归一化

class CLIPContrastiveLoss(nn.Module):
    def __init__(self,embed_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature  # 温度参数，控制对比学习的软硬程度

    def forward(self, img_queries, txt_queries):
        """
        计算 CLIP 对比学习损失。

        参数:
            img_queries: 图像查询向量，形状为 (B, N, D)
            txt_queries: 文本查询向量，形状为 (B, N, D)

        返回:
            对比学习损失（标量）
        """
        # 归一化查询向量
        img_queries = F.normalize(img_queries, p=2, dim=-1)  # 形状 (B, N, D)
        txt_queries = F.normalize(txt_queries, p=2, dim=-1)  # 形状 (B, N, D)

        # 计算相似度矩阵
        sim_matrix = torch.bmm(img_queries, txt_queries.transpose(1, 2))  # 形状 (B, N, N)
        sim_matrix = sim_matrix / self.temperature  # 缩放相似度

        # 构建对比目标
        batch_size, num_queries = img_queries.size(0), img_queries.size(1)
        labels = torch.arange(num_queries, device=img_queries.device)  # 正样本对的标签
        labels = labels.unsqueeze(0).expand(batch_size, -1)
        # 计算对比损失
        loss_img_txt = F.cross_entropy(sim_matrix, labels)  # 图像到文本的对比损失
        loss_txt_img = F.cross_entropy(sim_matrix.transpose(1, 2), labels)  # 文本到图像的对比损失

        # 总损失
        total_loss = (loss_img_txt + loss_txt_img) / 2
        return total_loss

class TextImageFusion(nn.Module):
    def __init__(self,embed_dim, num_heads, hidden_dim):
        super(TextImageFusion,self).__init__()
        self.attn=SelfAttention(embed_dim, num_heads=num_heads)
        self.sublayer_connect0=SublayerConnection(size=embed_dim)
        self.feed_forward=FeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.sublayer_connect1=SublayerConnection(size=embed_dim)

    def forward(self,text_hidden_states,image_hidden_states,attention_mask):
        #piexl=torch.cat([image_hidden_states,text_hidden_states],dim=1)
        text_image_attn=self.attn(text_hidden_states,image_hidden_states,image_hidden_states,attention_mask)
        text_image_attn=self.sublayer_connect0(x=text_hidden_states,sublayer_x=text_image_attn)
        text_image_fdd=self.feed_forward(text_image_attn)
        text_image_output=self.sublayer_connect1(x=text_image_attn,sublayer_x=text_image_fdd)
        return text_image_output

class SubTaskImg(nn.Module):
    def __init__(self,embed_dim,hidden_dim,num_attention_heads):
        super().__init__()
        self.multihead_attn = SelfAttention(embed_dim, num_heads=num_attention_heads)
        self.sublayer_connect1 = SublayerConnection(size=embed_dim)
        self.feed_forward = FeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.sublayer_connect2 = SublayerConnection(size=embed_dim)

    def forward(self,img_hidden_states,token_mask):
        img_output=self.multihead_attn(img_hidden_states,img_hidden_states,img_hidden_states,token_mask)
        img_output=self.sublayer_connect1(x=img_hidden_states,sublayer_x=img_output)
        img_output_feed=self.feed_forward(img_output)
        img_hidden=self.sublayer_connect2(x=img_output,sublayer_x=img_output_feed)
        return img_hidden



class TextOnlyMultiChannelGAT(nn.Module):
    def __init__(self, hidden_dim, num_heads, h=3, padding_idx=0, dropout_rate=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_heads = num_heads  # 多头注意力头数
        self.h = h  # 依赖矩阵的距离阈值数量（通道数）
        self.padding_idx = padding_idx  # 填充token的索引（用于掩码）
        self.dropout_rate = dropout_rate  # Dropout比率

        # 1. 多头自注意力机制（生成文本-文本注意力矩阵P^{t2t}）
        self.qk_proj = nn.Linear(hidden_dim, 2 * hidden_dim)  # Q、K投影

        # 2. 每个通道独立的GAT层（对应不同句法距离的依赖矩阵）
        self.gat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout_rate)  # 每个GAT层后添加Dropout
            )
            for _ in range(h)
        ])

        # 3. 添加Dropout层
        self.attention_dropout = nn.Dropout(dropout_rate)  # 注意力权重Dropout
        self.feature_dropout = nn.Dropout(dropout_rate)  # 特征Dropout

    def text_self_attention(self, Ht, attention_mask=None):
        # 计算文本-文本多头注意力矩阵P^{t2t}，加入掩码机制
        # Ht: [batch, seq_len, hidden_dim]
        # attention_mask: [batch, seq_len]，1表示有效token，0表示填充token
        batch_size, seq_len, _ = Ht.shape

        # 生成Q和K
        qk = self.qk_proj(Ht)  # [batch, seq_len, 2*hidden_dim]
        Q, K = torch.split(qk, self.hidden_dim, dim=-1)  # 拆分Q和K

        # 拆分多头
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # [batch, heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # [batch, heads, seq_len, d_k]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [batch, heads, seq_len, seq_len]

        # 应用掩码（填充token之间的注意力分数设为负无穷，softmax后为0）
        if attention_mask is not None:
            # 掩码形状调整：[batch, 1, 1, seq_len]（适配多头和序列维度）
            mask = (1 - attention_mask.unsqueeze(1).unsqueeze(1)) * -1e9
            scores = scores + mask  # [batch, heads, seq_len, seq_len]

        # 计算注意力权重（P^{t2t}）
        attn = F.softmax(scores, dim=-1)  # [batch, heads, seq_len, seq_len]

        # 对注意力权重应用Dropout（防止注意力过拟合）
        attn = self.attention_dropout(attn)

        return attn

    def forward(self, Ht, M_list, attention_mask=None):
        # Ht: 文本特征 [batch, n, hidden_dim]
        # M_list: 多通道依赖矩阵 [h个矩阵，每个为[batch, n, n]]
        # attention_mask: [batch, n]，1表示有效token，0表示填充token（可选）
        if attention_mask is not None:
            attention_mask = attention_mask.float()

        # 1. 生成带掩码的文本-文本注意力矩阵P^{t2t}
        P_t2t = self.text_self_attention(Ht, attention_mask)  # [batch, heads, n, n]

        # 2. 多通道特征计算（每个通道对应一个句法距离阈值）
        channel_features = []
        # if self.h==1:
        #     M_list=M_list.unsqueeze(0)
        # else:
        #     M_list = M_list.permute(1, 0, 2, 3)
        M_list = M_list.permute(1, 0, 2, 3)
        for i in range(self.h):
            M = M_list[i]  # 第i个通道的依赖矩阵 [batch, n, n]

            # 对依赖矩阵M应用掩码（过滤填充token的影响）
            if attention_mask is not None:
                # 生成[batch, n, n]的矩阵掩码（填充位置全为0）
                mask_matrix = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)  # [batch, n, n]
                M = M * mask_matrix  # 仅保留有效token之间的依赖关系

            # 文本-文本邻接矩阵：A^{t2t} = M ⊙ P^{t2t}（公式8）
            A_t2t = M.unsqueeze(1) * P_t2t  # 广播适配多头维度 [batch, heads, n, n]
            # 关键修改：对A_t2t的每行进行归一化（确保每行权重和为1）
            # 计算每行的和（避免除零，添加极小值epsilon）
            row_sums = A_t2t.sum(dim=-1, keepdim=True) + 1e-10  # [batch, heads, n, 1]
            A_t2t_normalized = A_t2t / row_sums  # 行归一化 [batch, heads, n, n]
            # GAT层聚合（公式9简化）
            feat_t2t = torch.matmul(A_t2t_normalized.mean(dim=1), Ht)  # 平均多头权重 [batch, n, hidden_dim]

            # 应用GAT层和ReLU激活
            feat_t2t = F.relu(self.gat_layers[i](feat_t2t))  # 第i通道独立处理
            #feat_t2t=self.gat_layers[i](feat_t2t)

            # 对特征应用Dropout（防止特征过拟合）
            feat_t2t = self.feature_dropout(feat_t2t)

            # 保留序列维度（不聚合），应用掩码（填充位置设为0）
            if attention_mask is not None:
                channel_feat = feat_t2t * attention_mask.unsqueeze(2)  # [batch, n, hidden_dim]
            else:
                channel_feat = feat_t2t  # [batch, n, hidden_dim]

            channel_features.append(channel_feat)

        # 3. 多通道特征融合（公式10），保持序列维度
        H_fusion = torch.stack(channel_features).mean(dim=0)  # [batch, n, hidden_dim]

        # 对融合特征应用最终Dropout
        H_fusion = self.feature_dropout(H_fusion)

        return H_fusion


class ImageOnlyHypergraphConv(nn.Module):
    """
    基于论文 DMR-XNet 的思想，为图像特征设计的超图卷积层。
    该层将图像的patch特征作为节点，通过特征相似度动态构建超图，并进行卷积。
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Args:
            input_dim (int): 输入特征的维度 (例如, ViT patch embedding dimension)。
            hidden_dim (int): 超图卷积中可学习权重矩阵的维度。
        """
        super(ImageOnlyHypergraphConv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 可学习的投影矩阵 W_HT-V (对应论文中的 W_l_HT-V)
        # 用于超图卷积操作
        self.W_conv = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # 初始化权重
        nn.init.xavier_uniform_(self.W_conv)

        # 论文中的全局特征感知增强参数 alpha
        # 这里设为可学习的参数，也可以设为固定值
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 线性投影矩阵 WV (对应论文中的 WV)
        # 将输入特征映射到公共特征空间
        self.WV = nn.Linear(input_dim, hidden_dim)

        # 可选：添加一个残差连接的门控或变换
        # 如果输入和输出维度不同，需要调整
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = None

    def global_feature_awareness_enhancement(self, V):
        """
        实现论文公式(1): 全局特征感知增强。
        增强每个patch特征的全局上下文感知能力。

        Args:
            V (Tensor): 形状为 (B, M, D) 的张量，B是批次大小，M是patch数量，D是特征维度。

        Returns:
            Tensor: 增强后的特征，形状为 (B, M, D)。
        """
        # 计算全局平均特征 (B, 1, D)
        global_feature = V.mean(dim=1, keepdim=True)  # shape: (B, 1, D)

        # 应用公式(1): v_i = alpha * v_i + (1 - alpha) * global_feature
        enhanced_V = self.alpha * V + (1 - self.alpha) * global_feature

        return enhanced_V

    def build_hypergraph_incidence_matrix(self, V_proj):
        """
        实现论文公式(6): 构建超图关联矩阵。
        基于特征相似度动态构建关联矩阵。

        Args:
            V_proj (Tensor): 投影后的图像特征，形状为 (B, M, H)。

        Returns:
            Tensor: 超图关联矩阵 H_T-V，形状为 (B, M, M)。
        """
        # 计算特征相似度矩阵 S_T-V (论文公式5)
        # V_proj: (B, M, H), V_proj.T: (B, H, M)
        # S_T-V: (B, M, M)
        similarity_matrix = torch.matmul(V_proj, V_proj.transpose(-1, -2))
        similarity_matrix = F.softmax(similarity_matrix, dim=-1)  # softmax along last dimension

        # 论文公式(6)中，对于单模态，我们可以简化为:
        # H_T-V = S_T-V (因为没有跨模态部分)
        # 在原文中，H_T-V 是一个 (2M, 2M) 的块矩阵，但因为我们只处理图像，
        # 我们直接使用相似度矩阵作为关联矩阵。
        # 注意：在原文中，H_T-V 是二分图的关联矩阵，但这里我们将其解释为
        # 节点-节点之间的连接强度，即一个完全连接的超图，每条边的权重由相似度决定。
        # 这是一种合理的单模态简化。
        H_TV = similarity_matrix  # shape: (B, M, M)

        return H_TV

    def hypergraph_convolution(self, X, H_TV):
        """
        实现论文公式(7): 超图卷积操作。

        Args:
            X (Tensor): 输入特征，形状为 (B, M, H)。
            H_TV (Tensor): 超图关联矩阵，形状为 (B, M, M)。

        Returns:
            Tensor: 卷积后的特征，形状为 (B, M, H)。
        """
        # 公式: O^l_T-V = H_TV * (H_TV^T * O^{l-1}_T-V * W^l_HT-V)

        # Step 1: H_TV^T * X
        # H_TV: (B, M, M), X: (B, M, H) -> (B, M, H)
        term1 = torch.matmul(H_TV.transpose(-1, -2), X)

        # Step 2: term1 * W_conv
        # term1: (B, M, H), W_conv: (H, H) -> (B, M, H)
        # 注意：这里 W_conv 是可学习的，需要广播到批次维度
        term2 = torch.matmul(term1, self.W_conv)

        # Step 3: H_TV * term2
        # H_TV: (B, M, M), term2: (B, M, H) -> (B, M, H)
        output = torch.matmul(H_TV, term2)

        return output

    def forward(self, V):
        """
        前向传播。

        Args:
            V (Tensor): 输入的图像patch特征，形状为 (B, M, D)。

        Returns:
            Tensor: 经过超图卷积和融合后的特征，形状为 (B, M, H)。
        """
        B, M, D = V.shape

        # 1. 全局特征感知增强 (论文公式1)
        V_enhanced = self.global_feature_awareness_enhancement(V)  # (B, M, D)

        # 2. 投影到公共特征空间 (论文公式3, 但只针对图像)
        V_proj = self.WV(V_enhanced)  # (B, M, H)

        # 3. 构建超图关联矩阵 (论文公式6)
        H_TV = self.build_hypergraph_incidence_matrix(V_proj)  # (B, M, M)

        # 4. 超图卷积 (论文公式7)
        # 初始输入 O^{l-1} 就是 V_proj
        O_conv = self.hypergraph_convolution(V_proj, H_TV)  # (B, M, H)

        # 5. 论文公式(9, 10)中的自适应融合
        # 这里简化为直接的残差连接
        # Tmul = Tproj + AdaptiveFusion(Tproj, O_conv)
        # 我们使用简单的加法和ReLU作为激活
        if self.residual_proj is not None:
            residual = self.residual_proj(V)  # (B, M, H)
        else:
            residual = V  # (B, M, D) -> 假设 D == H

        # 简单的融合: O_conv + residual, 然后通过激活函数
        # 这里可以替换为论文中的 AdaptiveFusion 函数
        fused_output = F.relu(O_conv + residual)

        return fused_output
# class ImageOnlyHypergraphConv(nn.Module):
#     """
#     基于论文 DMR-XNet 的思想，为图像特征设计的超图卷积层。
#     该层将图像的patch特征作为节点，通过特征相似度动态构建超图，并进行卷积。
#     添加了 Dropout 以增强正则化。
#     """
#
#     def __init__(self, input_dim, hidden_dim, dropout=0.1):
#         """
#         Args:
#             input_dim (int): 输入特征的维度 (例如, ViT patch embedding dimension)。
#             hidden_dim (int): 超图卷积中可学习权重矩阵的维度。
#             dropout (float): Dropout 概率。
#         """
#         super(ImageOnlyHypergraphConv, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#
#         # 可学习的投影矩阵 W_HT-V (对应论文中的 W_l_HT-V)
#         self.W_conv = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
#         nn.init.xavier_uniform_(self.W_conv)
#
#         # 全局特征感知增强参数 alpha
#         self.alpha = nn.Parameter(torch.tensor(0.5))
#
#         # 线性投影矩阵 WV
#         self.WV = nn.Linear(input_dim, hidden_dim)
#
#         # 添加 Dropout 层
#         self.dropout = nn.Dropout(dropout)
#
#         # 残差连接投影（如果维度不匹配）
#         if input_dim != hidden_dim:
#             self.residual_proj = nn.Linear(input_dim, hidden_dim)
#         else:
#             self.residual_proj = None
#
#     def global_feature_awareness_enhancement(self, V):
#         """
#         实现论文公式(1): 全局特征感知增强。
#         """
#         global_feature = V.mean(dim=1, keepdim=True)  # (B, 1, D)
#         enhanced_V = self.alpha * V + (1 - self.alpha) * global_feature
#         return enhanced_V
#
#     def build_hypergraph_incidence_matrix(self, V_proj):
#         """
#         构建超图关联矩阵 H_T-V。
#         """
#         # 计算相似度
#         similarity_matrix = torch.matmul(V_proj, V_proj.transpose(-1, -2))
#         similarity_matrix = F.softmax(similarity_matrix, dim=-1)
#         H_TV = similarity_matrix  # (B, M, M)
#         return H_TV
#
#     def hypergraph_convolution(self, X, H_TV):
#         """
#         超图卷积操作 O = H * (H^T * X * W)
#         """
#         term1 = torch.matmul(H_TV.transpose(-1, -2), X)  # H^T * X
#         term2 = torch.matmul(term1, self.W_conv)  # * W
#         output = torch.matmul(H_TV, term2)  # H * term2
#         return output
#
#     def forward(self, V):
#         """
#         前向传播。
#         """
#         B, M, D = V.shape
#
#         # 1. 全局特征感知增强
#         V_enhanced = self.global_feature_awareness_enhancement(V)  # (B, M, D)
#
#         # 2. 投影 + Dropout
#         V_proj = self.WV(V_enhanced)  # (B, M, H)
#         V_proj = self.dropout(V_proj)  # ✅ 在投影后加入 Dropout
#
#         # 3. 构建超图关联矩阵
#         H_TV = self.build_hypergraph_incidence_matrix(V_proj)  # (B, M, M)
#
#         # 4. 超图卷积
#         O_conv = self.hypergraph_convolution(V_proj, H_TV)  # (B, M, H)
#         O_conv = self.dropout(O_conv)  # ✅ 在卷积输出后加入 Dropout
#
#         # 5. 残差连接
#         if self.residual_proj is not None:
#             residual = self.residual_proj(V)  # (B, M, H)
#         else:
#             residual = V  # 假设 D == H
#
#         # 6. 融合 + 激活 + Dropout（可选）
#         fused_output = F.relu(O_conv + residual)
#         fused_output = self.dropout(fused_output)  # ✅ 可选：在最终输出前再加 Dropout
#
#         return fused_output