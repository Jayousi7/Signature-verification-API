import torch
from utils import set_seed
import torch.nn.functional as F
import torch.nn as nn
import copy 

class contrastive_loss(nn.Module):
    # Added measurement here so it's set once when you create the loss function
    def __init__(self, alpha=0.5, beta=0.5, m=0.7, measurement='euclidean'):
        super(contrastive_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.measurement = measurement
    
    def forward(self, output1, output2, Y):
        output1 = F.normalize(output1, p=2, dim=1)
        output2 = F.normalize(output2, p=2, dim=1)

        D = euclidean_distance(output1, output2)

        loss_similar = self.alpha * Y * torch.pow(D, 2)
        loss_dissimilar = self.beta * (1.0 - Y) * torch.pow(F.relu(self.m - D), 2)
        
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss

def euclidean_distance(output1, output2):
    distances = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-6)
    return distances


def train(model, device, train_data, test_data, epochs, ceritation, threshold, patience=3):
    set_seed()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,weight_decay=0.005)

    model = model.to(device)
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
 
        model.train()  
        running_train_loss = 0.0
        correct_train = 0.0
        total_train = 0.0        
        for batch_idx, (img1, img2, label) in enumerate(train_data):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            

            output1 = model(img1)
            output2 = model(img2)
            D = euclidean_distance(output1,output2)

            loss = ceritation(output1, output2, label)
            
            loss.backward()
            
            optimizer.step()
        

            predictions = (D < threshold).float()
            correct_train += (predictions == label).sum().item()
            total_train += label.size(0)            
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_data)
        train_acc = correct_train / total_train

        model.eval()  
        running_test_loss = 0.0
        correct_test =  0.0
        total_test = 0.0
        with torch.no_grad():

            for img1, img2, label in test_data:
                img1 = img1.to(device)
                img2 = img2.to(device)
                label = label.to(device)
                
                output1 = model(img1)
                output2 = model(img2)
                D =  euclidean_distance(output1,output2)
                loss = ceritation(output1, output2, label)
                
                running_test_loss += loss.item()
                predictions = (D < threshold).float()
                correct_test += (predictions == label).sum().item()
                total_test += label.size(0)

        avg_test_loss = running_test_loss / len(test_data)
        test_acc = correct_test / total_test

        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_weights)

    return model, history






