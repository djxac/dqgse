import logging
import os.path

from transformers import AutoConfig, TrainingArguments, Trainer, EvalPrediction, CLIPVisionModel, TrainerCallback
from transformers import BertForTokenClassification,RobertaForTokenClassification,AlbertForTokenClassification, ViTForImageClassification,SwinForImageClassification,DeiTModel, ConvNextForImageClassification
from model import DTCAModel
import torch

from model.model_valid import DTCAModel_valid
from utils.MyDataSet import  MyDataSet2
from utils.metrics import cal_f1, cal_f1_crf
import os
import argparse
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import Callable, Dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str,default='2017', nargs='?',help='display a string')
parser.add_argument('--task_name', type=str,default='valid', nargs='?',help='display a string')
parser.add_argument('--batch_size', type=int,default=8, nargs='?',help='display an integer')
parser.add_argument('--output_result_file', type=str,default="./result_15_valid.txt",nargs='?', help='display a string')
parser.add_argument('--output_dir', type=str,default="./results",nargs='?', help='display a string')
parser.add_argument('--lr', type=float, default=2e-6,nargs='?', help='display a float')
parser.add_argument('--epochs', type=int, default=3,nargs='?', help='display an integer')
parser.add_argument('--alpha', type=float, default=1.0,nargs='?', help='display a float')
parser.add_argument('--beta', type=float, default=1.0,nargs='?', help='display a float')
parser.add_argument('--text_model_name',type=str,default="roberta",nargs='?')
parser.add_argument('--image_model_name',type=str,default="vit",nargs='?')
parser.add_argument('--random_seed', type=int, default=1930,nargs='?')

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

def build_compute_metrics_fn(text_inputs, pairs, preds, threshold=0.4) -> Callable[[EvalPrediction], Dict]:
    if isinstance(pairs, torch.Tensor):
        true_labels = pairs.cpu().numpy().flatten()
    else:
        true_labels = np.array(pairs).flatten()

    if isinstance(preds, torch.Tensor):
        fallback_preds = preds.cpu().numpy().flatten()
    else:
        fallback_preds = np.array(preds).flatten()

    assert len(true_labels) == len(fallback_preds), "pairs 和 preds 长度必须一致"

    def compute_metrics_fn(p: EvalPrediction) -> Dict:
        cross_logits = p.predictions  

        probs = torch.nn.functional.softmax(torch.from_numpy(cross_logits), dim=-1).numpy()
        max_probs = probs.max(axis=-1)

        model_pred_labels = np.argmax(cross_logits, axis=-1)

        final_pred_labels = []
        for i, (conf, model_pred, fallback_pred) in enumerate(zip(max_probs, model_pred_labels, fallback_preds)):
            if conf <= threshold:
                final_pred_labels.append(fallback_pred)
            else:
                final_pred_labels.append(model_pred)

        final_pred_labels = np.array(final_pred_labels)

        accuracy = accuracy_score(true_labels, final_pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, final_pred_labels, average='macro'
        )

        if best_metric.get("f1") is None or f1 > best_metric["f1"]:
            best_metric.update({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "threshold": threshold,
                "fallback_rate": (max_probs < threshold).mean() 
            })

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fallback_rate": (max_probs < threshold).mean()
        }

    return compute_metrics_fn

# set random seed
set_random_seed(random_seed)

data_input_file = os.path.join("datasets/finetune",task_name,dataset_type,"input.pt")
data_inputs = torch.load(data_input_file)
train_word_ids = data_inputs["train"].word_ids
train_pairs = data_inputs["train"]["labels"]
train_dataset  = MyDataSet2(inputs=data_inputs["train"])


test_word_ids = data_inputs["test"].word_ids
test_pairs = data_inputs["test"]["labels"]
test_preds=data_inputs["test"]["preds"]
test_dataset  = MyDataSet2(inputs=data_inputs["test"])


# text pretrained model selected
if text_model_name == 'bert':
    model_path1 = './models/bert-base-uncased'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'roberta':
    model_path1 = "./roberta-base-cased"
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
    model_path2 = "./vit"
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
    model_path2 = "../clip-vit-base-patch32"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = CLIPVisionModel.from_pretrained(model_path2).state_dict()
else:
    os.error("出错了")
    exit()

# init DTCAModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vb_model = DTCAModel_valid(device,config1,config2,text_num_labels=5,text_model_name=text_model_name,image_model_name=image_model_name,alpha=alpha,beta=beta)
vb_model_dict = vb_model.state_dict()
for k,v in image_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
for k,v in text_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
vb_model.load_state_dict(vb_model_dict)

best_metric = dict()
text_best_metric = dict()

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_steps=10000,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    label_names=["labels"]
)

trainer = Trainer(
    model=vb_model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()

# output = trainer.predict(test_dataset=test_dataset)

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
