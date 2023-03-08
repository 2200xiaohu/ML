# 加载模型

https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2/InceptionResNetV2

[transformers官方](https://huggingface.co/docs/transformers/training)

[加载预训练模型，修改一些顶层](https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights)

[博客](https://towardsdatascience.com/the-most-favorable-pre-trained-sentiment-classifiers-in-python-9107c06442c6)

[博客2](https://www.vennify.ai/train-text-classification-transformer-model/)

[hugging face](https://huggingface.co/bert-base-uncased)

[博客3](http://fancyerii.github.io/2021/05/11/huggingface-transformers-1/)

[博客4](https://zhuanlan.zhihu.com/p/143209797)

# Hugging face

```
import torch
import torch.nn as nn
from transformers import BertModel

class BertCNN(nn.Module):
    def __init__(self, num_classes):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Convert to 1D convolutional format
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        x = self.conv1(last_hidden_state)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        # Apply max pooling and dropout
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        # Apply fully connected layer and softmax activation
        x = self.fc(x)
        x = self.softmax(x)
        return x

```



选取最后三层

```
import torch
import torch.nn as nn
from transformers import BertModel

class BertLast3Layers(nn.Module):
    def __init__(self, num_classes):
        super(BertLast3Layers, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(768*3, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        # Select last 3 layers and concatenate their hidden states
        last_3_layers = torch.cat((last_hidden_states[-3], last_hidden_states[-2], last_hidden_states[-1]), dim=-1)
        # Apply dropout and fully connected layer
        x = self.dropout(last_3_layers)
        x = self.fc(x)
        x = self.softmax(x)
        return x

```

