import numpy as np
import pandas as pd
import random
import torch.optim.optimizer
from torch.utils.tensorboard import SummaryWriter
from Ablation_ACP_CL import ProteinDataset, MyModel
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score


def compute_metrics(pred, label):
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    SN = tp / (tp + fn)
    SP = tn / (tn + fp)
    MCC = matthews_corrcoef(label, pred)
    ACC = accuracy_score(label, pred)
    AUC = roc_auc_score(label, pred)
    return ACC, MCC, SP, SN, AUC


def training(model, train_loader, optimizer, loss_cross, device):
    model.train()
    pred_list = []
    label_list = []
    num_batch = len(train_loader)
    train_loss = 0
    for FEGS, LSTM, label,samples in train_loader:
        input_ids = samples["input_ids"].to(device)
        token_type_ids = samples['token_type_ids'].to(device)
        attention_mask = samples['attention_mask'].to(device)
        label = label.to(device)
        FEGS = FEGS.to(device)
        LSTM = LSTM.to(device)
        optimizer.zero_grad()

        logit = model(FEGS, LSTM, input_ids,token_type_ids,attention_mask)
        label = label.to(dtype=torch.float32)

        pred = logit.squeeze()
        cross_loss = loss_cross(pred, label)

        pred = pred > 0.5
        train_loss += cross_loss

        cross_loss.backward()
        optimizer.step()

        pred_list.extend(pred.cpu().detach().numpy())
        label_list.extend(label.cpu().detach().numpy())
    ACC, _, _, _, _ = compute_metrics(pred_list, label_list)

    return ACC, train_loss / num_batch


def validing(model, valid_loader, loss_cross, device):
    model.eval()
    pred_list = []
    label_list = []
    num_batch = len(valid_loader)
    valid_loss = 0

    with torch.no_grad():
        for FEGS, LSTM, label, samples in valid_loader:
            input_ids = samples["input_ids"].to(device)
            token_type_ids = samples['token_type_ids'].to(device)
            attention_mask = samples['attention_mask'].to(device)
            label = label.to(device)
            FEGS = FEGS.to(device)
            LSTM = LSTM.to(device)

            logit = model(FEGS, LSTM, input_ids, token_type_ids, attention_mask)
            label = label.to(dtype=torch.float32)
            pred = logit.squeeze()

            cross_loss = loss_cross(pred, label)
            valid_loss += cross_loss
            pred = pred > 0.5
            pred_list.extend(pred.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())
        ACC, MCC, SP, SN, AUC = compute_metrics(pred_list, label_list)
        return ACC, MCC, SP, SN, AUC, pred_list, label_list, valid_loss / num_batch


if __name__ == "__main__":
    #ACP740
    data = pd.read_csv("./data/acp740.csv")
    seq = data.iloc[:, 0]
    seqence = []
    for i in seq:
       	seqence.append(" ".join(i))
    seq = np.array(seqence)
    label = data.iloc[:, 1]
    label = np.array(label)
    FEGS = torch.load("./data/FEGS_ACP740.pt")
    LSTM = torch.load("./data/LSTM_ACP740.pt")
    train_seq, test_seq, train_label, test_label, train_FEGS, test_FEGS, train_LSTM, test_LSTM = train_test_split(seq,
                                                                                                                   label,
                                                                                                                   FEGS,
                                                                                                                   LSTM,
                                                                                                                   test_size=0.2,
                                                                                                                   random_state=42)
    
    test_dataset = ProteinDataset(test_seq, test_label, test_FEGS, test_LSTM)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, drop_last=True)

    # #ACP-main
    # np.random.seed(66)
    # torch.manual_seed(66)
    # random.seed(66)
    # torch.cuda.manual_seed_all(66)
    # torch.backends.cudnn.deterministic = True
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # train = pd.read_csv("./data/ACP-main/ACP2.0-main.csv")
    # train_seq = train.iloc[:, 0]
    # train_seq = np.array(train_seq)
    # seqence = []
    # for i in train_seq:
    #     seqence.append(" ".join(i))
    # train_seq = np.array(seqence)
    # train_label = train.iloc[:, 1]
    # train_label = np.array(train_label)
    # train_FEGS = torch.load("./data/ACP-main/FEGS_ACP2.0-main.pt")
    # train_LSTM = torch.load("./data/ACP-main/LSTM_ACP2.0-main.pt")
    #
    # test = pd.read_csv("./data/ACP-main/ACP2.0-main_test.csv")
    # test_seq = test.iloc[:, 0]
    # test_seq = np.array(test_seq)
    # seqence = []
    # for i in test_seq:
    #     seqence.append(" ".join(i))
    # test_seq = np.array(seqence)
    # test_label = test.iloc[:, 1]
    # test_label = np.array(test_label)
    # test_FEGS = torch.load("./data/ACP-main/FEGS_ACP2.0-main_test.pt")
    # test_LSTM = torch.load("./data/ACP-main/LSTM_ACP2.0-main_test.pt")
    # test_dataset = ProteinDataset(test_seq, test_label, test_FEGS, test_LSTM)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=66)
    best_parameter = 0
    best_ACC = 0.0
    for i in range(50, 51):
        ACC_score = []
        MCC_score = []
        SP_score = []
        SN_score = []
        AUC_score = []
        for index, (train_idx, val_idx) in enumerate(skf.split(train_seq, train_label)):
            print("**" * 10, "the", index + 1, "fold", "ing...", "**" * 10)
            train_x, valid_x = train_seq[train_idx],train_seq[val_idx]
            train_y, valid_y = train_label[train_idx], train_label[val_idx]
            train_FEGS_, valid_FEGS = train_FEGS[train_idx], train_FEGS[val_idx]
            train_LSTM_, valid_LSTM = train_LSTM[train_idx], train_LSTM[val_idx]

            train_dataset = ProteinDataset(train_x, train_y, train_FEGS_, train_LSTM_)
            valid_dataset = ProteinDataset(valid_x, valid_y, valid_FEGS, valid_LSTM)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
            valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, drop_last=True)

            model = MyModel()
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_cross = torch.nn.BCELoss()


            for epoch in range(i):
                train_ACC, train_l = training(model, train_loader, optimizer, loss_cross, device)
                valid_ACC, MCC, SP, SN, AUC, _, _, valid_loss = validing(model, valid_loader, loss_cross, device)
                print(f'Epoch:{epoch}  Train Score{train_ACC}   Vailid Score {valid_ACC}')


            print("Test")
            test_ACC, test_MCC, test_SP, test_SN, test_AUC, pred_list, label_list, valid_loss = validing(model,
                                                                                                         test_loader,
                                                                                                         loss_cross,
                                                                                                         device)
            print(f'ACC:{test_ACC},MCC:{test_MCC},SP:{test_SP},SN:{test_SN},AUC:{test_AUC}')
            ACC_score.append(test_ACC)
            MCC_score.append(test_MCC)
            SP_score.append(test_SP)
            SN_score.append(test_SN)
            AUC_score.append(test_AUC)
        print("Average")
        print(
            f"ACC:{sum(ACC_score) / len(ACC_score)},MCC:{sum(MCC_score) / len(MCC_score)}, SP:{sum(SP_score) / len(SP_score)}, SN:{sum(SN_score) / len(SN_score)}, AUC:{sum(AUC_score) / len(AUC_score)}")
        if sum(ACC_score) / len(ACC_score) >= best_ACC:
            best_parameter = i
            best_ACC = sum(ACC_score) / len(ACC_score)

    print("over")
    print(f"best_parameter:{best_parameter}___bestACC:{best_ACC}")
