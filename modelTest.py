import h5py
import pandas as pd
from tqdm import tqdm

from fastai.vision.all import *
from utils_data import add_ageGroup, get_patient_label

from utils_data import split_data, MILBagTransform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch.optim as optim
from utils_model import MILModel
from utils_train import train_model, test_model


cls = ['epithelium', 'stroma'] # use embeddings of these specific class
model_name = 'UNI' # define to use which model's embeddings to train MIL
rootdir = '/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/RootDir/KHP_RM' # containing .h5 features
clinic = '/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/DATA/KHP_clinic.csv' # containing age group

folds = 5 # define how many folds
resFolder = f'/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/experiments/KHP_RM/{model_name}' # save patient's fold assigning results
if not os.path.exists(resFolder):
    os.mkdir(resFolder)


# input wsi->patch->patch embeddings->bag data tensor
bag_size = 512 # how many patches sampled or padded for each slide

# dataloader
batch_size = 4 # how many patients/slides to load each step

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU for training
dim_input = 1024 # UNI's input dim ???
dim_output = 3 # output prediction

# optimiser
lr = 0.0001 # optimiser's lr
weight_decay = 0.00001 # optimiser's regularity

# early_stopping
patience = 5
minEpochTrain = 10
max_epochs = 20


clinic_df = pd.read_csv(clinic) # get patients' info
clinic_df = add_ageGroup(clinic_df) # add ground truth age group
patientID, yTrue, yTrueLabel = get_patient_label(clinic_df, rootdir, model_name) # categorical ->numeric labels
kf = StratifiedKFold(n_splits=folds, random_state=0, shuffle=True) # initialise KF
kf.get_n_splits(patientID, yTrue)

foldcounter = 0
for train_index, test_index in kf.split(patientID, yTrue):
    print(f'This is fold {foldcounter}')
    print('-'*30)
    print('splitting data')
    print('-'*30)
    fold_pt = f'{resFolder}/{model_name}_fold{foldcounter}_train.csv'
    train_data, val_data, test_data = split_data(clinic_df, rootdir, patientID, yTrue, train_index, test_index, fold_pt) # .h5 file paths added

    
    print('creating dataloaders (path->tensor)')
    print('-'*30)
    dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                   get_x = ColReader('feaBag_pt'),
                   get_y = ColReader('age_group'),
                   splitter = ColSplitter('is_valid'),
                   item_tfms = MILBagTransform(train_data.feaBag_pt, bag_size, cls))
    dls = dblock.dataloaders(train_data, bs = batch_size)
    trainLoaders = dls.train
    valLoaders = dls.valid

    
    print('preparing attention-based MIL model...')
    print('-'*30)
    model = MILModel(dim_input, dim_output).to(device)


    print('preparing weighted loss function...')
    print('-'*30)
    weight = train_data['age_group'].value_counts().sum() / train_data['age_group'].value_counts()
    weight /= weight.sum()
    weight = torch.tensor(list(map(weight.get, dls.vocab)))
    criterion = CrossEntropyLossFlat(weight = weight.to(torch.float32)) # weighted loss function to handle imbalanced classes
    criterion.to(device)

    
    print('preparing Adam optimizer')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    
    print('-' * 30)
    print('START TRAINING ...')
    print('-' * 30)

    model, train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(model, trainLoaders, valLoaders, 
                                                                                                  optimizer, criterion,
                                                                                                  patience, minEpochTrain, max_epochs, 
                                                                                                  model_name, foldcounter,resFolder)
  
    torch.save(model.state_dict(), f'{resFolder}/{model_name}_MILfinalModel_fold{foldcounter}')
    history = pd.DataFrame({'train_loss': train_loss_history, 'train_acc': train_acc_history,
                           'val_loss': val_loss_history, 'val_acc': val_acc_history})
    history.to_csv(f'{resFolder}/{model_name}_TrainValhistory_fold{foldcounter}.csv')

    print('-' * 30)
    print('START TESTING ...')
    print('-' * 30)

    model.load_state_dict(torch.load(f"{resFolder}/{model_name}_MILbestModel_fold{foldcounter}"))
    model = model.to(device)
    testdl = dls.test_dl(test_data)

    probsList = test_model(model=model, testLoaders=testdl)

    probs = {}
    for key in ['<30y', '30-50y', '>50y']:
        probs[key] = []
        for item in probsList:
            probs[key].append(item[targetLabelDict[key]])
            
    probs = pd.DataFrame.from_dict(probs)
    test_data = test_data.rename({'cls': 'yTrueLabel'})
    test_data['yTrue'] = [targetLabelDict[i] for i in test_data['yTrueLabel']]
    testResults = pd.concat([test_data, probs], axis = 1) 
    testResults_pt = f'{resFolder}/{model_name}_testResult_perSlide_fold(foldcounter).csv'
    testResults.to_csv(testResultsPath, index = False)




    









