import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn import CrossEntropyLoss, LayerNorm
import torchvision
import numpy as np
import os
from transformers import RobertaModel, BertModel, AlbertModel, ElectraModel, ViTModel, SwinModel, DeiTModel, \
    ConvNextModel, CLIPVisionModel

from model.query_model import TextImageFusionModel, CLIPContrastiveLoss, TextImageFusion, TextOnlyMultiChannelGAT, \
    SubTaskImg, FeedForward, SublayerConnection, CrossAttention, ImageOnlyHypergraphConv


class DTCAModel(nn.Module):
    def __init__(self,device,config1,config2,text_num_labels,alpha,beta,text_model_name="roberta",image_model_name='vit'):
        super().__init__()
        if text_model_name == 'roberta':
            self.roberta = RobertaModel(config1,add_pooling_layer=False)
        elif text_model_name == 'bert':
            self.bert = BertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'albert':
            self.albert = AlbertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'electra':
            self.electra = ElectraModel(config1)
        if image_model_name == 'vit':
            self.vit = ViTModel(config2)
        elif image_model_name == 'swin':
            self.swin = SwinModel(config2)
        elif image_model_name == 'deit':
            self.deit = DeiTModel(config2)
        elif image_model_name == 'convnext':
            self.convnext = ConvNextModel(config2)
        elif image_model_name == 'clip':
            # 对于CLIP模型，使用其视觉部分的配置
            from transformers import CLIPVisionConfig

            # 从完整配置中提取视觉配置
            if hasattr(config2, 'vision_config'):
                clip_vision_config = CLIPVisionConfig.from_dict(config2.vision_config.to_dict())
            else:
                clip_vision_config = config2  # 作为后备方案

            # 初始化CLIP视觉模型
            self.clip_model = CLIPVisionModel(clip_vision_config)

            # 冻结CLIP模型参数
            for param in self.clip_model.parameters():
                param.requires_grad = False
        self.alpha = alpha
        self.beta = beta
        self.text_model_name=text_model_name
        self.image_model_name=image_model_name
        self.config1 = config1
        self.config2 = config2
        self.text_num_labels = text_num_labels

        #self.reshape=ReshapeSequence(197,60,768)
        self.query_model=TextImageFusionModel(device,None,1,96,768,768,8,2048,None)

        self.image_text_cross = MultiHeadAttention(8,config1.hidden_size,config1.hidden_size,config1.hidden_size)

        self.dropout = nn.Dropout(0.1)
        self.loss_fct = CrossEntropyLoss()

        self.classifier1 = nn.Linear(config1.hidden_size, self.text_num_labels)
        self.classifier0= nn.Linear(config1.hidden_size,self.text_num_labels)
        self.CRF = CRF(self.text_num_labels,batch_first=True)
        self.contract_loss=CLIPContrastiveLoss(768)

        self.gcn = TextOnlyMultiChannelGAT(768, 4,2)
        self.layer_norm=LayerNorm(768)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                image_labels=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None,dependency_matrics=None,noun_masks=None,cap_input_ids=None,cap_attention_mask=None):
        return_dict = return_dict if return_dict is not None else self.config1.use_return_dict
        if self.text_model_name == 'bert':
            text_outputs = self.bert(input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        elif self.text_model_name == 'roberta':
            text_outputs = self.roberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'albert':
            text_outputs = self.albert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'electra':
            text_outputs = self.electra(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        else:
            text_outputs=None
        if self.image_model_name == 'vit':
            image_outputs = self.vit(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'swin':
            image_outputs = self.swin(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'deit':
            image_outputs = self.deit(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'convnext':
            image_outputs = self.convnext(pixel_values)
        elif self.image_model_name == 'clip':
            image_outputs = self.clip_model(pixel_values)
        else:
            image_outputs=None


        text_last_hidden_states = text_outputs["last_hidden_state"]

        image_last_hidden_states = image_outputs["last_hidden_state"]

        image_attention_mask=torch.ones(image_last_hidden_states.size(0),image_last_hidden_states.size(1), dtype=torch.long,device=image_last_hidden_states.device)
        tgt,text_queries,image_queries= self.query_model(text_last_hidden_states,attention_mask,image_last_hidden_states,image_attention_mask,cap_input_ids,cap_attention_mask)
        contract_loss=self.contract_loss(image_queries,text_queries)
        text_hidden_states_n=torch.mul(noun_masks.unsqueeze(2),text_last_hidden_states)
        noun_hidden=text_hidden_states_n.sum(dim=1)/(torch.sum(noun_masks,dim=1,keepdim=True) + 1e-8)
        noun_loss=self.cosine_similarity_loss(torch.mean(text_queries+image_queries,dim=1),noun_hidden)

        # cross_crf_loss
        text_hidden_states_syn=self.layer_norm(self.dropout(self.gcn(text_last_hidden_states,dependency_matrics,attention_mask))+text_last_hidden_states)

        image_text_cross_attention, _ = self.image_text_cross(text_hidden_states_syn,tgt,tgt)


        cross_logits = self.classifier0(image_text_cross_attention)

        mask = (labels != -100)
        mask[:,0] = 1
        # print(cross_logits.shape, cross_labels.shape)
        cross_crf_loss =  -self.CRF(cross_logits,cross_labels,mask=mask) / 10

        # text_loss
        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)
        
        # getTextLoss: CrossEntropy
        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))

        loss = cross_crf_loss + 0.3*text_loss + contract_loss +noun_loss
        # 将list转换为numpy数组（方便后续处理）
        # end train
        return {"loss":loss,
            "logits":text_token_logits,
            "cross_logits": cross_logits
                }

    def pad_lists_with_minus_one(self,lists):
        # 找到最长子列表的长度
        max_length = max(len(lst) for lst in lists) if lists else 0
        # 对每个子列表填充-1至max_length
        padded_lists = []
        for lst in lists:
            # 计算需要填充的长度
            pad_length = max_length - len(lst)
            # 填充-1并添加到结果中
            padded_lst = lst + [-1] * pad_length
            padded_lists.append(padded_lst)
        return padded_lists
    def cosine_similarity_loss(self,x, y):
        """
        计算两个向量的余弦相似度损失

        参数:
        x, y: 输入向量，形状为 [batch_size, feature_dim]

        返回:
        loss: 余弦相似度损失，标量
        """
        # 对向量进行 L2 归一化
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)

        # 计算余弦相似度（batch 中每个样本的相似度）
        cos_sim = torch.sum(x_norm * y_norm, dim=1)

        # 计算损失：1 - 余弦相似度的平均值
        loss = 1 - cos_sim.mean()

        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE损失计算，适用于图文匹配任务
    同一对图文作为正样本，批次内其他图文对作为负样本
    """

    def __init__(self, temperature=0.07):
        """
        参数:
            temperature: 温度参数，控制分布的陡峭程度
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        """
        计算InfoNCE损失

        参数:
            image_embeddings: 图像的嵌入向量，形状为 [batch_size, embedding_dim]
            text_embeddings: 文本的嵌入向量，形状为 [batch_size, embedding_dim]

        返回:
            loss: 计算得到的InfoNCE损失
        """
        batch_size = image_embeddings.size(0)

        # 归一化嵌入向量
        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

        # 计算图像和文本之间的相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())

        # 对角线元素是正样本对的相似度
        # 每个样本有1个正样本和(batch_size-1)个负样本
        positive_similarities = torch.diag(similarity_matrix)

        # 计算所有样本对的相似度除以温度
        similarity_matrix /= self.temperature

        # 计算分母：所有样本的相似度指数和（包括正样本）
        exp_similarities = torch.exp(similarity_matrix)
        sum_exp = torch.sum(exp_similarities, dim=1)

        # 计算每个样本的损失
        per_sample_loss = -torch.log(
            torch.exp(positive_similarities / self.temperature) / sum_exp
        )

        # 计算批次的平均损失
        loss = torch.mean(per_sample_loss)

        return loss
class ReshapeSequence(nn.Module):
    def __init__(self, seq_len, new_seq_len, embed_dim):
        super().__init__()
        self.proj = nn.Linear(seq_len, new_seq_len)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)  # -> (batch_size, embed_dim, seq_len)
        x = self.proj(x)       # -> (batch_size, embed_dim, new_seq_len)
        x = x.transpose(1, 2)  # -> (batch_size, new_seq_len, embed_dim)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss



def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device
                     ).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(
        b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device
                       ) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2)/beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad,
                           beta=0.5, iteration=50, k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
             beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance


def kl_divergence_loss(p_logits, q_logits, mask=None):
    '''
    计算概率分布 p 和 q 之间的KL散度，自动对齐输入序列长度。
    :param p_logits: [N, *, L1] 原始分布的 logits。
    :param q_logits: [N, *, L2] 目标分布的 logits。
    :param mask: [N] 可选的权重或掩码，用于指示有效样本。
    '''
    # 确保输入张量的序列长度维度一致
    # 获取两个张量的序列长度（假设倒数第二维是序列长度维度）
    p_seq_len = p_logits.size(-2)
    q_seq_len = q_logits.size(-2)

    # 如果长度不一致，则进行插值对齐
    if p_seq_len != q_seq_len:
        # 选择目标长度为较长的序列长度
        target_seq_len = max(p_seq_len, q_seq_len)

        # 调整p_logits的长度
        if p_seq_len < target_seq_len:
            # 转换形状为适合插值的维度 (N, *, features, seq_len)
            p_reshaped = p_logits.transpose(-2, -1)
            # 线性插值
            p_interpolated = F.interpolate(
                p_reshaped,
                size=target_seq_len,
                mode='linear',
                align_corners=False
            )
            # 恢复原始维度顺序
            p_logits = p_interpolated.transpose(-2, -1)

        # 调整q_logits的长度
        if q_seq_len < target_seq_len:
            q_reshaped = q_logits.transpose(-2, -1)
            q_interpolated = F.interpolate(
                q_reshaped,
                size=target_seq_len,
                mode='linear',
                align_corners=False
            )
            q_logits = q_interpolated.transpose(-2, -1)

    # 计算概率分布
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    # 计算KL散度，不使用reduction参数以保持输出形状
    kl_div = F.kl_div(p_log_probs, q_probs, reduction='none')

    # 在样本维度进行求和
    kl_div = kl_div.sum(dim=-1).sum(dim=-1)

    # 应用mask以仅选择有效的样本
    if mask is not None:
        mask = mask.to(dtype=kl_div.dtype)
        mask = mask.unsqueeze(-1)  # 扩展mask的形状为[N, 1]使其能够与kl_div进行广播
        kl_div = kl_div * mask

        # 计算使用mask加权平均后的损失
        kl_div_loss = kl_div.sum() / mask.sum()
    else:
        # 如果没有mask，则直接计算平均损失
        kl_div_loss = kl_div.mean()

    return kl_div_loss



