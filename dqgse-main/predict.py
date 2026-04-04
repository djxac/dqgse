import logging
import os.path

from transformers import AutoConfig, TrainingArguments, Trainer, EvalPrediction, CLIPVisionModel, TrainerCallback
from transformers import BertForTokenClassification,RobertaForTokenClassification,AlbertForTokenClassification, ViTForImageClassification,SwinForImageClassification,DeiTModel, ConvNextForImageClassification
from model import DTCAModel
import torch
from utils.MyDataSet import  MyDataSet2
from utils.metrics import cal_f1, cal_f1_crf
from typing import Callable, Dict
import numpy as np
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str,default='2017', nargs='?',help='display a string')
parser.add_argument('--task_name', type=str,default='dualc', nargs='?',help='display a string')
parser.add_argument('--batch_size', type=int,default=4, nargs='?',help='display an integer')
parser.add_argument('--output_result_file', type=str,default="./result_17.txt",nargs='?', help='display a string')
parser.add_argument('--output_dir', type=str,default="./results",nargs='?', help='display a string')
parser.add_argument('--lr', type=float, default=2e-5,nargs='?', help='display a float')
parser.add_argument('--epochs', type=int, default=40,nargs='?', help='display an integer')
parser.add_argument('--alpha', type=float, default=1.0,nargs='?', help='display a float')
parser.add_argument('--beta', type=float, default=1.0,nargs='?', help='display a float')
parser.add_argument('--text_model_name',type=str,default="roberta",nargs='?')
parser.add_argument('--image_model_name',type=str,default="vit",nargs='?')
parser.add_argument('--random_seed', type=int, default=2940,nargs='?')

args = parser.parse_args()
dataset_type = args.dataset_type
task_name = args.task_name
alpha = args.alpha
beta = args.beta
batch_size = args.batch_size
output_dir = args.output_dir
lr = args.lr
epochs = args.epochs
text_model_name = args.text_model_name
image_model_name = args.image_model_name
output_result_file = args.output_result_file
random_seed = args.random_seed


def set_random_seed(random_seed):
    """Set random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True


def predict(p_dataset, p_inputs, p_pairs):
    outputs = trainer.predict(p_dataset)
    pred_labels = np.argmax(outputs.predictions[0], -1)
    return cal_f1(pred_labels,p_inputs,p_pairs)


def build_compute_metrics_fn(text_inputs,pairs) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        text_logits, cross_logits= p.predictions
        #image_attn_weights=torch.from_numpy(image_attn_weights)
        #torch.save(image_attn_weights,"./text_attn_weights_17.pt")
        text_pred_labels = np.argmax(text_logits,-1)
        pred_labels = np.argmax(cross_logits,-1)

        precision, recall, f1 = cal_f1(pred_labels,text_inputs,pairs,False)

        text_precision, text_recall, text_f1 = cal_f1(text_pred_labels, text_inputs, pairs)
        if best_metric.get("f1") is not None:
            if f1 > best_metric["f1"]:
                best_metric["f1"] = f1
                best_metric["precision"] = precision
                best_metric["recall"] = recall
                # with open("my_model_result.txt", "w", encoding="utf-8") as f:
                #     f.write(str(pred_labels.tolist())+ '\n')
        else:
            best_metric["f1"] = f1
            best_metric["precision"] = precision
            best_metric["recall"] = recall
            # with open("my_model_result.txt", "w", encoding="utf-8") as f:
            #     f.write(str(pred_labels.tolist())+ '\n')
        if text_best_metric.get("f1") is not None:
            if text_f1 > text_best_metric["f1"]:
                text_best_metric["f1"] = text_f1
                text_best_metric["precision"] = text_precision
                text_best_metric["recall"] = text_recall
        else:
            text_best_metric["f1"] = text_f1
            text_best_metric["precision"] = text_precision
            text_best_metric["recall"] = text_recall
        return {"precision": precision,"recall":recall, "f1": f1}
    return compute_metrics_fn

# set random seed
set_random_seed(random_seed)

data_input_file = os.path.join("utils/datasets/finetune",task_name,dataset_type,"input_syn.pt")
data_inputs = torch.load(data_input_file)
train_word_ids = data_inputs["train"].word_ids
train_pairs = data_inputs["train"]["pairs"]
data_inputs["train"].pop("pairs")
train_dataset  = MyDataSet2(inputs=data_inputs["train"])

dev_word_ids = data_inputs["dev"].word_ids
dev_pairs = data_inputs["dev"]["pairs"]
data_inputs["dev"].pop("pairs")
dev_dataset  = MyDataSet2(inputs=data_inputs["dev"])

test_word_ids = data_inputs["test"].word_ids
test_pairs = data_inputs["test"]["pairs"]
data_inputs["test"].pop("pairs")
test_dataset  = MyDataSet2(inputs=data_inputs["test"])

# text pretrained model selected
if text_model_name == 'bert':
    model_path1 = './models/bert-base-uncased'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'roberta':
    model_path1 = "../lqmasa/roberta-base-cased"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'albert':
    model_path1 = "./models/albert-base-v2"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'electra':
    model_path1 = './models/electra-base-discriminator'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1).state_dict()
else:
    os.error("出错了")
    exit()

# image pretrained model selected
if image_model_name == 'vit':
    model_path2 = "vit"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'swin':
    model_path2 = "./models/swin-tiny-patch4-window7-224"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'deit':
    model_path2 = "./models/deit-base-patch16-224"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = DeiTModel.from_pretrained(model_path2).state_dict()
elif image_model_name == 'convnext':
    model_path2 = './models/convnext-tiny-224'
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ConvNextForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'clip':
    # CLIP视觉编码器
    model_path2 = "../lqmasa/clip-vit-base-patch32"  # 可以根据需要更换其他CLIP模型
    config2 = AutoConfig.from_pretrained(model_path2)
    # 使用CLIP的视觉模型部分
    image_pretrained_dict = CLIPVisionModel.from_pretrained(model_path2).state_dict()
else:
    os.error("出错了")
    exit()

# init DTCAModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vb_model = DTCAModel(device,config1,config2,text_num_labels=5,text_model_name=text_model_name,image_model_name=image_model_name,alpha=alpha,beta=beta)
vb_model_dict = vb_model.state_dict()

checkpoint_path = "./autodl/best_model_17.pth/pytorch_model.bin"

# 加载检查点
vb_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# load pretrained model weights
# for k,v in image_pretrained_dict.items():
#     if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
#         vb_model_dict[k] = v
# for k,v in text_pretrained_dict.items():
#     if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
#         vb_model_dict[k] = v
# vb_model.load_state_dict(vb_model_dict)
# vb_model.query_model.queries.load_state_dict(checkpoint['query_model']['queries'])
# vb_model.query_model.layers[0].load_state_dict(checkpoint['query_model']['layers'])
best_metric = dict()
text_best_metric = dict()

training_args = TrainingArguments(
    output_dir=output_dir,
    #evaluation_strategy="epoch",
    save_steps=10000,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    label_names=["labels","cross_labels"]
)

# 自定义回调函数保持不变
class MultiDatasetEvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset, test_compute_fn,val_dataset,val_compute_fn):
        self.test_dataset = test_dataset
        self.test_compute_fn = test_compute_fn
        self.val_dataset = val_dataset
        self.val_compute_fn = val_compute_fn
        self.last_evaluated_epoch = -1  # 记录上一次评估的 epoch，避免重复

    def on_epoch_end(self, args, state, control, **kwargs):
        # 仅在完整 epoch 结束时评估测试集，且每个 epoch 只评估一次
        if state.epoch is not None and int(state.epoch) > self.last_evaluated_epoch:
            # 保存当前评估函数
            #current_compute_fn = trainer.compute_metrics
            # 临时替换为测试集评估函数
            trainer.compute_metrics = self.val_compute_fn
            # 评估测试集
            val_metrics = trainer.evaluate(
                eval_dataset=self.val_dataset,
                metric_key_prefix="val"
            )
            #恢复原来的评估函数
            trainer.compute_metrics = self.test_compute_fn
            test_metrics = trainer.evaluate(
                eval_dataset=self.test_dataset,
                metric_key_prefix="test"
            )
            # #
            # if(state.epoch == 36):
            #     trainer.save_model('./best_model_15.pth')
            # 更新上一次评估的 epoch
            self.last_evaluated_epoch = int(state.epoch)

trainer = Trainer(
    model=vb_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=build_compute_metrics_fn(text_inputs=data_inputs['test'],pairs=test_pairs),
)
#trainer.train()

# output = trainer.predict(test_dataset=test_dataset)
trainer.evaluate(
                eval_dataset=test_dataset,
                metric_key_prefix="test"
            )
# save results
with open(output_result_file,"a",encoding="utf-8") as f:
    model_para = dict()
    model_para["dataset_type"] = dataset_type
    model_para["text_model"] = text_model_name
    model_para["image_model"] = image_model_name
    model_para["batch_size"] = batch_size
    model_para["alpha"] = alpha
    model_para["beta"] = beta
    model_para['random_seed']=random_seed
    f.write("参数:"+str(model_para) + "\n")
    f.write("multi: "+ str(best_metric)+"\n")
    f.write("text: "+ str(text_best_metric)+"\n")
    f.write("\n")