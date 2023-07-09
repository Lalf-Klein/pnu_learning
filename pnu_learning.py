import torch
import numpy as np
from mydataset import MyDataset


class PNULearning:
    """
    PNULearning class as described in the paper
    Tomoya Sakai, Marthinus Christoffel du Plessis, Gang Niu, and Masashi Sugiyama. 
    Semi-supervised classification based on classification from positive and unlabeled data. 
    In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, pp.2998–3006, 2017.
    
    Args:
        model (torch.nn.Module): Deep learning model with Torch
        loss_func: Loss functions like "torch.sigmoid"
        optimizer: torch.optim.Optimizer like "torch.optim.Adam"
        lr (float): Learning rate
        p_ratio (float): Percentage of positive data
        eta (float): Hyperparameters that take the values between 0 and 1
        device (str): CPU/GPU information obtained by "torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
    """

    def __init__(self, model, loss_func, optimizer, lr, p_ratio, eta, device):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.p_ratio = p_ratio
        self.eta = eta
        self.device = device

    
    def t_P_index(self, t):
        return torch.maximum(t, torch.zeros_like(t))
    
    def t_N_index(self, t):
        return torch.maximum(-t, torch.zeros_like(t))
    
    def t_U_index(self, t):
        return torch.ones_like(t) - torch.abs(t)
    
    def Risk(self, index, y):
        n = torch.max(torch.tensor([1, torch.sum(index).item()]))
        k = torch.sum(torch.mul(index, self.loss_func(-y)))
        return torch.div(k, n)
    
    def Risk_P_plus(self, t, y):
        return self.Risk(self.t_P_index(t), y)
    
    def Risk_P_minus(self, t, y):
        return self.Risk(self.t_P_index(t), -y)
    
    def Risk_N_plus(self, t, y):
        return self.Risk(self.t_N_index(t), y)
    
    def Risk_N_minus(self, t, y):
        return self.Risk(self.t_N_index(t), -y)
    
    def Risk_U_plus(self, t, y):
        return self.Risk(self.t_U_index(t), y)
    
    def Risk_U_minus(self, t, y):
        return self.Risk(self.t_U_index(t), -y)
    
    def Risk_PN(self, t, y):
        return self.p_ratio * self.Risk_P_plus(t, y) + (1-self.p_ratio) * self.Risk_N_minus(t, y)
    
    def Risk_PU(self, t, y):
        return self.p_ratio * (self.Risk_P_plus(t, y) - self.Risk_P_minus(t, y)) + self.Risk_U_minus(t, y)
        
    def Risk_NU(self, t, y):
        return (1 - self.p_ratio) * (self.Risk_N_minus(t, y) - self.Risk_N_plus(t, y)) + self.Risk_U_plus(t, y)
    
    def Risk_PNU(self, t, y):
        t = t.flatten()
        y = y.flatten()
        if self.eta >= 0:
            return (1 - self.eta) * self.Risk_PN(t, y) + self.eta * self.Risk_PU(t, y)
        else:
            return (1 + self.eta) * self.Risk_PN(t, y) - self.eta * self.Risk_NU(t, y)
    
    def accuracy(self, t, y):
        t = t.flatten()
        y = y.flatten()
        y = y[t!=0]
        t = t[t!=0]
        return torch.mean(torch.maximum(torch.sign(y) * t, torch.zeros_like(t))).item() 
    
    def recall(self, t, y, threshold):
        t = t.flatten()
        y = y.flatten()
        y = y[t==1]
        t = t[t==1]
        if len(t) > 0:
            p = len(y[y>threshold])/len(t)
        else:
            p = np.nan
        return p
    
    def precision(self, t, y, threshold):
        t = t.flatten()
        y = y.flatten()        
        y = y[t!=0]
        t = t[t!=0]        
        t = t[y>threshold]
        y = y[y>threshold]
        if len(y) > 0:
            r = len(t[t==1])/len(y)
        else:
            r = np.nan
        return r  
    
    def fit(self, x_train, y_train, x_test, y_test, n_epoch, batch_size, threshold=0, verbose=1):
        self.model.to(self.device)
        train_dataset = MyDataset(x_train, y_train)
        test_dataset = MyDataset(x_test, y_test)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        train_loss_history = []
        train_acc_history = []
        train_precision_history = []
        train_recall_history = []
        test_loss_history = []
        test_acc_history = []
        test_precision_history = []
        test_recall_history = []
        
        for epoch in range(n_epoch):
            print("epoch: {epoch}/{n_epoch}".format(epoch=epoch+1, n_epoch=n_epoch))
            total_loss = 0
            temp_acc_hist = []
            temp_precision_hist = []
            temp_recall_hist = []
            #train_loop
            self.model.train()
            for batch, (X, t) in enumerate(train_dataloader):
                X = X.to(self.device)
                t = t.to(self.device)
                pred = self.model(X)
                loss = self.Risk_PNU(t, pred)
                acc = self.accuracy(t, pred)
                precision = self.precision(t, pred, threshold)
                recall = self.recall(t, pred, threshold)
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                if acc >= 0 and acc <= 1:
                    temp_acc_hist.append(acc)
                if precision >= 0 and precision <= 1:
                    temp_precision_hist.append(precision)
                if recall >= 0 and recall <= 1:
                    temp_recall_hist.append(recall)
                if verbose > 0 and batch % verbose == 0 and batch != 0:
                    print("loss: {loss:.4f}, acc: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}".format(loss=total_loss/(batch+1), 
                                                                                                      acc = np.mean(np.array(temp_acc_hist)[~np.isnan(temp_acc_hist)]),
                                                                                                      precision=np.mean(np.array(temp_precision_hist)[~np.isnan(temp_precision_hist)]), 
                                                                                                      recall=np.mean(np.array(temp_recall_hist)[~np.isnan(temp_recall_hist)])))

            train_loss_history.append(total_loss/(batch+1))
            train_acc_history.append(np.mean(np.array(temp_acc_hist)[~np.isnan(temp_acc_hist)]))
            train_precision_history.append(np.mean(np.array(temp_precision_hist)[~np.isnan(temp_precision_hist)]))
            train_recall_history.append(np.mean(np.array(temp_recall_hist)[~np.isnan(temp_recall_hist)]))
            
            #test_loop
            self.model.eval()
            with torch.no_grad():
                test_total_loss = 0
                test_temp_acc_hist = []
                test_temp_precision_hist = []
                test_temp_recall_hist = []
                for batch, (X, t) in enumerate(test_dataloader):
                    X = X.to(self.device)
                    t = t.to(self.device)
                    pred = self.model(X)
                    test_total_loss += self.Risk_PNU(t, pred).item()
                    test_temp_acc_hist.append(self.accuracy(t, pred))
                    test_temp_precision_hist.append(self.precision(t, pred, threshold))
                    test_temp_recall_hist.append(self.recall(t, pred, threshold))
            print("test_loss: {loss:.4f}, test_acc: {acc:.4f}, test_precision: {precision:.4f}, test_recall: {recall:.4f}".format(loss=test_total_loss/(batch+1), 
                                                                                                                  acc=np.mean(np.array(test_temp_acc_hist)[~np.isnan(test_temp_acc_hist)]), 
                                                                                                                  precision=np.mean(np.array(test_temp_precision_hist)[~np.isnan(test_temp_precision_hist)]), 
                                                                                                                  recall=np.mean(np.array(test_temp_recall_hist)[~np.isnan(test_temp_recall_hist)])))
            test_loss_history.append(test_total_loss/(batch+1))
            test_acc_history.append(np.mean(np.array(test_temp_acc_hist)[~np.isnan(test_temp_acc_hist)]))
            test_precision_history.append(np.mean(np.array(test_temp_precision_hist)[~np.isnan(test_temp_precision_hist)]))
            test_recall_history.append(np.mean(np.array(test_temp_recall_hist)[~np.isnan(test_temp_recall_hist)]))            
        
        return train_loss_history, train_acc_history, train_precision_history, train_recall_history, \
               test_loss_history, test_acc_history, test_precision_history, test_recall_history
    
    def predict(self, x_test, threshold=0):
        x_test
        _t = np.zeros(len(x_test))
        pred_dataset = MyDataset(x_test, _t)
        test_dataloader = torch.utils.data.DataLoader(pred_dataset, batch_size=len(pred_dataset)//5, shuffle=False)
        self.model.eval()
        pred = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for batch, (X, _t) in enumerate(test_dataloader):
                X = X.to(self.device)
                temp_pred = self.model(X).flatten()
                temp_pred[temp_pred>=threshold] = 1
                temp_pred[temp_pred<threshold] = -1
                pred = torch.cat((pred, temp_pred))
        return pred
    
