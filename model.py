import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from transformers import BertModel, AutoTokenizer

model_path = './bert'
tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)


class MyModel(nn.Module):
    def __init__(self, input_dim_left=578, input_dim_right=20, out_channels_left=8, kernel_size_left=16,
                 kernel_size_middel=3, stride_middle=1, dropout_sample=0.15, dropout_positive=0.3,
                 dropout_negative=0.9):
        super(MyModel, self).__init__()
        # left
        self.out_channels_left_ = out_channels_left
        self.kernel_size_left_ = kernel_size_left
        self.kernel_size_middel_ = kernel_size_middel
        self.stride_middle_ = stride_middle
        self.conv1_left = nn.Conv1d(in_channels=1, out_channels=self.out_channels_left_, kernel_size=self.kernel_size_left_)
        conv_length = input_dim_left - self.kernel_size_left_ + 1
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.line_left = nn.Linear(self.out_channels_left_ * conv_length, 512)
        self.dropout_left = nn.Dropout(0.5)

        # bert
        self.bert = BertModel.from_pretrained(model_path)
        for name, param in self.bert.named_parameters():
            if 'encoder.layer' in name:
                layer_num = int(name.split('.')[2])
                if layer_num >= self.bert.config.num_hidden_layers - 4:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        self.conv1_middle = nn.Conv1d(1024, 512, kernel_size=self.kernel_size_middel_, stride=self.stride_middle_,padding="same")
        self.line_middle = nn.Linear(512 * 50, 512)
        self.dropout_middle = nn.Dropout(0.5)

        # right
        self.bilstm = nn.LSTM(input_size=input_dim_right, hidden_size=256,
                              bidirectional=True, batch_first=True)
        self.Layer_norm = nn.LayerNorm(512)
        self.attention = Attention(512)


        # contrastive layer
        self.infonce_loss = infonce.InfoNCE()
        self.contrastive_dropout1 = nn.Dropout(dropout_sample)
        self.contrastive_dropout2 = nn.Dropout(dropout_positive)
        self.contrastive_dropout3 = nn.Dropout(dropout_negative)

        # # Final layers--add
        self.dense_final_1 = nn.Linear(512 * 3, 256)
        self.dropout_final = nn.Dropout(0.5)
        self.dense_final_2 = nn.Linear(256, 16)
        self.dense_final_3 = nn.Linear(16, 1)

    def forward(self, input_left, input_right, input_ids, token_type_ids, attention_mask):
        # input_left: [batch_size,578]
        x_left = input_left.unsqueeze(1)
        x_left = self.conv1_left(x_left)
        x_left = self.relu(x_left)
        x_left = self.flatten(x_left)
        x_left = self.line_left(x_left)
        x_left = self.dropout_left(x_left)

        # input_middle : [batch_size,50,1024]
        x_middle, _ = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        x_middle = x_middle.permute(0, 2, 1)
        x_middle = self.conv1_middle(x_middle)
        x_middle = x_middle.permute(0, 2, 1)
        x_middle = self.attention(x_middle)

        # input_right: [batch_size, 50, 32]
        x_right, _ = self.bilstm(input_right)  # [batch_size,50, 512]
        forward = x_right[:, -1, :256]
        backward = x_right[:, 0, 256:]
        x_right = torch.cat((forward, backward), dim=1)
        x_right = self.Layer_norm(x_right)  # [batch_size, 512]

        # Concatenate and final layers
        x_final = torch.cat([x_left, x_right, x_middle], dim=1)

        x = F.relu(self.dense_final_1(x_final))
        x = self.dropout_final(x)
        x = F.relu(self.dense_final_2(x))
        x = self.dropout_final(x)
        x = self.dense_final_3(x)
        # x = self.dropout_final(x)

        return nn.Sigmoid()(x)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, output):
        weights = torch.softmax(self.attention_weights(output), dim=1)
        weighted_output = torch.sum(output * weights, dim=1)
        return weighted_output


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, train, labels, Left, Right):
        self.train = train
        self.labels = labels
        self.left = Left
        self.lstm = Right

    def __getitem__(self, idx):
        seq = self.train[idx]
        encoding = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=50,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding="max_length",
            return_tensors='pt',
        )
        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        label = self.labels[idx]
        LEFT = self.left[idx, :578]
        LSTM = self.lstm[idx]
        LEFT = torch.tensor(LEFT, dtype=torch.float32)
        LSTM = torch.tensor(LSTM, dtype=torch.float32)

        return LEFT, LSTM, label, sample

    def __len__(self):
        return len(self.labels)
