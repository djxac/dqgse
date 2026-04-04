from transformers import AutoConfig,TrainingArguments,Trainer,EvalPrediction,BertForTokenClassification,ViTForImageClassification
from typing import Callable, Dict


def cal_f1(p_pred_labels,p_inputs,p_pairs,is_result=False):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    pred_pair_list = []
    true_pair_list=[]
    for i, pred_label in enumerate(p_pred_labels):
        word_ids = p_inputs.word_ids(batch_index=i)
        flag = False
        pred_pair = set()
        sentiment = 0
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if word_ids[j] is None:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
                continue
            if word_ids[j] != word_ids[j - 1]:
                if pp > 1:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    start_pos = word_ids[j]
                    end_pos = word_ids[j]
                    sentiment = pp - 2
                    flag = True
                elif pp == 1:
                    if flag:
                        end_pos = word_ids[j]
                else:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
        true_pair = set(p_pairs[i])
        gold_num += len(true_pair)
        predict_num += len(list(pred_pair))
        #print(pred_pair)
        pred_pair_list.append(pred_pair.copy())
        true_pair_list.append(true_pair.copy())
        correct_num += len(true_pair & pred_pair)
    precision = 0
    recall = 0
    f1 = 0
    if predict_num != 0:
        precision = correct_num / predict_num
    if gold_num != 0:
        recall = correct_num / gold_num
    if precision != 0 or recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    if is_result:
        #save_pred_pair_list(pred_pair_list,'pred_pair_test.txt')
        save_pred_pair_list(true_pair_list, 'pred_pair_15_true.txt')
        #print(correct_num)
        return precision*100, recall*100, f1*100
    else:
        return precision*100, recall*100, f1*100


def save_pred_pair_list(pred_pair_list, save_path):
    """
    将pred_pair_list保存到txt文件

    参数：
        pred_pair_list: 模型预测的实体对列表（列表套集合套元组）
        save_path: 保存路径（例如"./pred_pairs.txt"）
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx, pred_set in enumerate(pred_pair_list):
            # 每个样本的预测结果单独占一行，行首添加样本索引（方便对应原始数据）
            # 将集合转换为列表，再将每个元组格式化为字符串（如"(0-2, 0)"）
            pair_strs = [f"({pos},{senti})" for (pos, senti) in pred_set]
            # 用逗号拼接同一行的所有预测对，例如："0: (0-2,0),(3-5,1)"
            line = f"{idx}: " + ",".join(pair_strs) + "\n"
            f.write(line)
    print(f"pred_pair_list已保存到：{save_path}")

def cal_f1_crf(p_pred_labels,p_inputs,p_pairs,is_result=False):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    pred_pair_list = []
    for i, pred_label in enumerate(p_pred_labels):
        flag = False
        pred_pair = set()
        sentiment = 0
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if j ==0:
                continue
            if pp > 1:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                start_pos = j-1
                end_pos = j-1
                sentiment = pp - 2
                flag = True
            elif pp == 1:
                if flag:
                    end_pos = j-1
            else:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                flag = False
        if flag:
            pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
        true_pair = set(p_pairs[i])
        gold_num += len(true_pair)
        predict_num += len(list(pred_pair))
        pred_pair_list.append(pred_pair.copy())
        correct_num += len(true_pair & pred_pair)
    precision = 0
    recall = 0
    f1 = 0
    if predict_num != 0:
        precision = correct_num / predict_num
    if gold_num != 0:
        recall = correct_num / gold_num
    if precision != 0 or recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    if is_result:
        return precision*100, recall*100, f1*100,pred_pair_list
    else:
        return precision*100, recall*100, f1*100

def cal_single_f1(p_pred_labels,p_inputs,p_pairs,is_result=False):
    gold_num = 0
    predict_num = 0
    correct_num = 0
    pred_pair_list = []
    for i, pred_label in enumerate(p_pred_labels):
        word_ids = p_inputs.word_ids(batch_index=i)
        flag = False
        pred_pair = set()
        sentiment = 0
        start_pos = 0
        end_pos = 0
        for j, pp in enumerate(pred_label):
            if word_ids[j] is None:
                if flag:
                    pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
                continue

            if word_ids[j] != word_ids[j - 1]:
                if 0<pp<4:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    start_pos = word_ids[j]
                    end_pos = word_ids[j]
                    sentiment = pp - 1
                    flag = True
                elif pp == sentiment + 4:
                    if flag:
                        end_pos = word_ids[j]
                else:
                    if flag:
                        pred_pair.add((str(start_pos) + "-" + str(end_pos), sentiment))
                    flag = False
        true_pair = set(p_pairs[i])
        gold_num += len(true_pair)
        predict_num += len(pred_pair)
        pred_pair_list.append(pred_pair.copy())
        correct_num += len(true_pair & pred_pair)

    precision = 0
    recall = 0
    f1 = 0
    if predict_num != 0:
        precision = correct_num / predict_num
    if gold_num != 0:
        recall = correct_num / gold_num
    if precision != 0 or recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    if is_result:
        return precision * 100, recall * 100, f1 * 100, pred_pair_list
    else:
        return precision * 100, recall * 100, f1 * 100


