import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer,AdamW
import torch.nn.functional as F
from split_label import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import os
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from random import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score

text=load_vectors('./data.txt')
# keyword = load_vectors_keyword('./keywords_data.txt')
# text['keyword'] = keyword['keyword']
#移除空值
text=text.drop([i for i in range(0,len(text)) if len(text['text'][i])<4])

SEED=2011
#text=text.loc[text.index!=460]#出去label10，因为只有一个
y=text['label']
le=preprocessing.LabelEncoder()
y=le.fit_transform(y)
x = text.drop('label',axis=1)

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=SEED)
x_train, x_val, y_train ,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=SEED)
x_train.reset_index(inplace=True,drop=True)
x_val.reset_index(inplace=True,drop=True)
x_test.reset_index(inplace=True,drop=True)
train = pd.DataFrame({'text':x_train['text'],'label':y_train}).reset_index(drop=True)
val = pd.DataFrame({'text':x_val['text'],'label':y_val}).reset_index(drop=True) 
test = pd.DataFrame({'text':x_test['text'],'label':y_test}).reset_index(drop=True)

train = pd.read_csv("./bert_insert-1.csv")

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.loc[index, 'text'])
        label = self.data.loc[index, 'label']

        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            return_token_type_ids=False, 
            pad_to_max_length=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )

        return {
#             'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# train_data = pd.read_csv('train.csv')
train_data = train
train_dataset = CustomDataset(train_data, tokenizer, max_len=512)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_data = val
val_dataset = CustomDataset(val_data, tokenizer, max_len=512)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

test_data = test
test_dataset = CustomDataset(test_data, tokenizer, max_len=512)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

time_point = 10

def train(model, device, loss_fn,train_loader, optimizer, epoch):
        model.train()
        train_loss, correct = 0, 0
#         train_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc="Training", dynamic_ncols=True, position=0)
        train_pbar = enumerate(train_dataloader)
        true_labels = []
        predicted_labels = []
        print(f'Epoch {epoch}')
        
        for step, batch in train_pbar:
            batch = tuple(t.to(device) for t in batch.values())
            input_ids, attention_mask, labels = batch
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#             print(f'output siza is {outputs.size()}')
            loss = loss_fn(outputs, labels)
            train_loss +=loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(pred.squeeze().cpu().numpy())

            
            # Backward pass
            loss.backward()
            optimizer.step()
            
#         description = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, step * len(batch), len(train_loader.dataset),
#                 100. * step / len(train_loader), loss.item())
#         train_pbar.set_postfix(description)
#             train_pbar.set_postfix(loss=loss.item(),learning_rate=optimizer.param_groups[0]['lr'])
        train_loss /= len(train_loader.dataset)
    
        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
        macro_precision = precision_score(true_labels, predicted_labels, average='macro')
        macro_recall = recall_score(true_labels, predicted_labels, average='macro')
        micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    
    
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Macro_f1: {:.4f}, Micro_f1: {:.4f}, macro_precision: {:.4f}, macro_recall: {:.4f}\n'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset),macro_f1,micro_f1,macro_precision,macro_recall))
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Lr/train",optimizer.param_groups[0]['lr'],epoch)
        writer.add_scalar("Accuracy/train",100. * correct / len(train_loader.dataset),epoch)
        writer.add_scalar("F1_macro/train", macro_f1, epoch)
        writer.add_scalar("Precision_macro/train", macro_precision, epoch)
        writer.add_scalar("Recall_macro/train", macro_recall, epoch)
        writer.add_scalar("F1_micro/train", micro_f1, epoch)

def test(model, device, test_loader , loss_fn,epoch):
    model.eval()
    test_loss = 0
    correct = 0
#     val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader),desc="Validation", dynamic_ncols=True, position=0)
    val_pbar = enumerate(val_dataloader)
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for step, batch in val_pbar:
            batch = tuple(t.to(device) for t in batch.values())
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            test_loss += loss_fn(outputs, labels).item()  # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(pred.squeeze().cpu().numpy())
            
    test_loss /= len(test_loader.dataset)
    
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    macro_precision = precision_score(true_labels, predicted_labels, average='macro')
    macro_recall = recall_score(true_labels, predicted_labels, average='macro')
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Macro_f1: {:.4f}, Micro_f1: {:.4f}, macro_precision: {:.4f}, macro_recall: {:.4f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),macro_f1,micro_f1,macro_precision,macro_recall))
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("Accuracy/test",100. * correct / len(test_loader.dataset),epoch)
    writer.add_scalar("F1_macro/test", macro_f1, epoch)
    writer.add_scalar("Precision_macro/test", macro_precision, epoch)
    writer.add_scalar("Recall_macro/test", macro_recall, epoch)
    writer.add_scalar("F1_micro/test", micro_f1, epoch)
    
    #Save model
    if(epoch%time_point==0):
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_Average loss': test_loss,
        'val_acc': correct / len(test_loader.dataset)
        }, f'./checkpoint/checkpoint_epoch_{epoch}.pt')
    torch.cuda.empty_cache()
    return test_loss


class BertClassifier(nn.Module):
    def __init__(self, num_classes, freeze_bert=True):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-large-uncased')
        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.freeze_bert = freeze_bert
        self.unfreeze_layers = 4 #解冻
        # set Freeze BERT layers
        if self.freeze_bert:
            num_layers = self.bert.config.num_hidden_layers
            unfreeze_start_layer = num_layers - self.unfreeze_layers
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
                if 'encoder' in name:
                    layer_index = int(name.split('.')[2])
                    if layer_index >= unfreeze_start_layer:
                        param.requires_grad = True
        self.dropout = nn.Dropout(0.1)
        # self.linear1 = nn.Linear(self.bert.config.hidden_size, 128)
        # self.linear2 = nn.Linear(128, num_classes)
        # self.Classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size, 256), self.dropout ,nn.Linear(256, 128),nn.Linear(128, num_classes))
        # self.Classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size*2), nn.ReLU(),nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size),self.dropout,nn.Linear(self.bert.config.hidden_size, num_classes))
        self.Classifier = nn.Sequential(nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size), nn.Linear(self.bert.config.hidden_size, num_classes))
    #hidden_size=768
    #outputs.last_hidden_state
    #outputs.attentions
    #outputs.hidden_states
    
    def reset_parameters(self):
        for name, module in self.named_modules():
            if 'encoder' not in name and 'embeddings' not in name and 'pooler' not in name:#除去bert部分
                if isinstance(module, nn.Linear):
                    print(f'reset the {name}')
                    nn.init.xavier_uniform_(module.weight.data)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.LayerNorm):
                    print(f'reset the {name}')
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask):
        
        #最后一层第一个位置即CLS
        #outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True)
        # pooled = outputs.hidden_states[-1][:,0,:]
        
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled = outputs.pooler_output
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True)
        cls_output = self.max_pool(outputs)
      
        
        x = self.dropout(cls_output)
        x = self.Classifier(x)
        
        return x
    
    def get_optimizer(self, learning_rate, weight_decay):
        # Only parameters that require gradients
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.bert.named_parameters() if p.requires_grad]},
            {"params": [p for n, p in model.named_parameters() if 'encoder' not in n and 'embeddings' not in n and 'pooler' not in n and p.requires_grad],
             "lr": learning_rate*10, "weight_decay": weight_decay},
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        return optimizer
    
    def check_freeze(self):
         for name, module in self.named_parameters():
                if module.requires_grad:
                    print(name)
    
    def concat(self,outputs):
        last_four_layers = outputs.hidden_states[-4:]
        pooled = torch.cat(last_four_layers, dim=-1)
        cls_output = pooled[:, 0, :]# size:(batch_size, 4 * hidden_size)
        return cls_output
    
    def max_pool(self,outputs):
        last_four_layers = outputs.hidden_states[-4:]
        pooled_outputs = []

        for layer in last_four_layers:
            max_pool_output, _ = torch.max(layer, dim=1)
            pooled_outputs.append(max_pool_output)
            
        concatenated_outputs = torch.cat(pooled_outputs, dim=1)
        return concatenated_outputs
    
    def mean_pool(self,outputs):
        last_four_layers = outputs.hidden_states[-4:]
        pooled_outputs = []

        for layer in last_four_layers:
            mean_pool_output = torch.mean(layer, dim=1)
            pooled_outputs.append(mean_pool_output)

        # Concatenate the pooled outputs
        concatenated_outputs = torch.cat(pooled_outputs, dim=1)
        return concatenated_outputs
        

# 为不同的层设置不同的学习率
def get_layer_params(layer, learning_rate):
    param_optimizer = list(layer.named_parameters())
    layer_params = [{"params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                     "weight_decay": 0.01, "lr": learning_rate},
                    {"params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                     "weight_decay": 0.0, "lr": learning_rate}]
    return layer_params

#调整层间学习率
def change_lr(model,lr):
    other_params = [
        {'params': [p for n,p in model.Classifier.named_parameters()], 'lr': lr*10},
    ]
    pooler_params = [{'params': [p for n,p in model.bert.pooler.named_parameters() if p.requires_grad], 'lr': lr}]
    for name, module in model.named_modules():
        if not 'encoder' in name and not 'embeddings' in name and not 'pooler' in name:
            if isinstance(module, nn.LayerNorm):
                print(f'set lr for LayerNorm on {name}')
                other_params.append({'params': module.parameters(), 'lr': lr*10})
    params = []
    params.extend(other_params)
    if len(pooler_params[-1]['params'])>0:
        params.extend(pooler_params)
    for i in reversed(range(len(model.bert.encoder.layer))):
        layer = model.bert.encoder.layer[i]
        layer_params = get_layer_params(layer, lr)
        lr*=0.9
        if len(layer_params[0]['params'])>0:
            params.extend(layer_params)
    
    return params


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertClassifier(num_classes=9)
model.reset_parameters()#初始化
model.check_freeze()

#-----------------------------加载预训练bert----------------#
checkpoint = torch.load("model_InDomain_61.pth")
current_epoch = checkpoint["epoch"]

#load params
pretrained_dict = checkpoint["model"]

#获取当前网络的dict
model_state_dict = model.state_dict()

#剔除不匹配的权值参数
pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}

#更新新模型参数字典
model_state_dict.update(pretrained_dict_1)

#将包含预训练模型参数的字典"放"到新模型中
model.load_state_dict(model_state_dict)

#将模型移到所有可用的GPU上
if torch.cuda.device_count() > 1:
    print("使用 %d 个GPU。" % torch.cuda.device_count())
    model = nn.DataParallel(model)


model.to(device)

lr_base = 1e-5
weight_decay = 0.1

params = change_lr(model,lr_base)
params.reverse()

from torch.utils.tensorboard import SummaryWriter 
writer = SummaryWriter('/root/tf-logs')  

loss_fn = nn.CrossEntropyLoss()
# optimizer = model.get_optimizer(learning_rate=1e-5, weight_decay=0.1)
optimizer = AdamW(params, betas=(0.9, 0.999), weight_decay=weight_decay,lr=lr_base*10,eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, cooldown=3,factor=0.2,min_lr=5e-9)


for epoch in range(1, 71):
#     if(epoch%time_point==0):
#         model = BertClassifier(num_classes=9).to(device)
#         loss_fn = nn.CrossEntropyLoss()
#         optimizer = model.get_optimizer(learning_rate=1e-5, weight_decay=0.1)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, cooldown=3,factor=0.2,min_lr=5e-9)
    train(model, optimizer=optimizer,loss_fn=loss_fn, train_loader = train_dataloader,epoch=epoch,device=device)
    val_loss=test(model,device=device,test_loader=val_dataloader,loss_fn = loss_fn,epoch=epoch)
    scheduler.step(val_loss)
writer.close()
os.system("/usr/bin/shutdown")

# cd ~/autodl-tmp
#python after_ft.py > train.log 2>&1 tail -f train.log
