import h5py
import pandas as pd
from fastai.vision.all import *
from tqdm import tqdm


def get_dim_input(model_name):
    if model_name == 'UNI':
        dim_input = 1024
    elif model_name == 'iBOT':
        dim_input = 768
    elif model_name == 'gigapath':
        dim_input = 1536
    elif model_name == 'ResNet50':
        dim_input = 2048
    return dim_input
    
    
class EarlyStopping:
    def __init__(self, patience = 20, stop_epoch = 50, verbose=False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'\nEarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss    
        




def train_model(model, trainLoaders, valLoaders, 
                optimizer, criterion,
                patience, minEpochTrain, max_epochs, 
                model_name, stainFunc, foldcounter,resFolder):
    since = time.time()
    
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    

    early_stopping = EarlyStopping(patience=patience, stop_epoch=minEpochTrain, verbose=True)
    

    for epoch in range(max_epochs):
        print(f"{epoch}/{max_epochs-1} epoch")
        print('\nTraining\n')
        phase = 'train'
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs,labels in tqdm(trainLoaders): # inputs = (emeddings, patch lens)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs, attention_scores = model(inputs)
                loss = criterion(outputs, labels)
                _, y_hat = torch.max(outputs, 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(y_hat == labels.data)
                
        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.item() / len(trainLoaders.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        print(type(epoch_loss), type(epoch_acc))

        print(f'\n {phase} Loss: {np.round(epoch_loss, 4)}  ACC: {np.round(epoch_acc, 4)}')

        if valLoaders:
            print('Validation....\n')
            phase = 'val'
            model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in tqdm(valLoaders):
                with torch.set_grad_enabled(phase=='train'):
                    outputs, attention_scores = model(inputs)
                    loss = criterion(outputs, labels)
                    _, y_hat = torch.max(outputs, dim=1)
                    running_loss += loss.item() * inputs[0].size(0)
                    running_corrects += torch.sum(y_hat == labels.data)
                    
            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.item() / len(valLoaders.dataset)
            val_loss_history.append(val_loss)     
            val_acc_history.append(val_acc) 
            
            print(f'\n {phase} Loss: {np.round(val_loss, 4)}   ACC: {np.round(val_acc, 4)}')

            ckpt_name = f"{resFolder}/{model_name}_{stainFunc}_MILbestModel_fold{foldcounter}.pt"

            early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)
            if early_stopping.early_stop:
                print('-'*30)
                print(f'Training stop, the validation loss did not drop for {patience} epochs!!!')
                print('-'*30)
                break
            print('-' * 30)

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    
    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history






def test_model(model, dataloaders):
    phase = 'test'
    model.eval()
    probsList = []
    for inputs in tqdm(dataloaders):
        with torch.set_grad_enabled(phase=='train'):
            outputs, attention_scores = model(inputs[0])
            probs = nn.Softmax(dim=1)(outputs)
            probsList = probsList + probs.tolist()
    return probsList, attention_scores





