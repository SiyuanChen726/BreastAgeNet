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
from utils_train import get_dim_input, train_model, test_model




cls = ['epithelium', 'stroma'] # use embeddings of these specific class
rootdir = '/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/RootDir/SGK_healthy' # containing .h5 features
clinic = '/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/DATA/clinicData/SGK_2k.csv' # containing age group


# -- use features extracted by {model_name} and processed by {stainFunc} for training -- #
model_name = 'iBOT' 
stainFunc = 'augmentation'
batch_size = 4 # how many patients/slides to load each step



folds = 5 # define how many folds
resFolder = f'/scratch/prj/cb_normalbreast/Siyuan/prj_normal/BreastAgeNet/experiments/SGK_healthy/{model_name}_{stainFunc}' # save patient's fold assigning results
if not os.path.exists(resFolder):
    os.mkdir(resFolder)


# loader
bag_size = 512 # how many patches sampled or padded for each slide


# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU for training
dim_input = get_dim_input(model_name) # UNI's input dim 
dim_output = 3 # output prediction


# optimiser
lr = 0.0001 # optimiser's lr
weight_decay = 0.00001 # optimiser's regularity


# early_stopping
patience = 5
minEpochTrain = 20
max_epochs = 100


# data split
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
    fold_pt = f'{resFolder}/{model_name}_{stainFunc}_fold{foldcounter}_train.csv'
    train_data, val_data, test_data = split_data(clinic_df, rootdir, stainFunc, patientID, yTrue, train_index, test_index, fold_pt) # .h5 file paths added
    print('train_data')
    print(train_data['age_group'].value_counts())
    print('val_data')
    print(val_data['age_group'].value_counts())
    print('test_data')
    print(test_data['age_group'].value_counts())

    
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
    model = MILModel(dim_input, dim_output, with_attention_scores=True).to(device)


    print('preparing weighted loss function...')
    print('-'*30)
    weight = train_data['age_group'].value_counts().sum() / train_data['age_group'].value_counts()
    weight /= weight.sum()
    weight = torch.tensor(list(map(weight.get, dls.vocab)))
    criterion = CrossEntropyLossFlat(weight = weight.to(torch.float32)) # weighted loss function to handle imbalanced classes
    criterion.to(device)


    print('preparing optimizer')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)


    print('-' * 30)
    print('START TRAINING ...')
    print('-' * 30)
    model, train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(model, trainLoaders, valLoaders, 
                                                                                                      optimizer, criterion,
                                                                                                      patience, minEpochTrain, max_epochs, 
                                                                                                      model_name, stainFunc, foldcounter,resFolder)



    torch.save(model.state_dict(), f'{resFolder}/{model_name}_{stainFunc}_MILfinalModel_fold{foldcounter}.pt')
    history = pd.DataFrame({'train_loss': train_loss_history, 'train_acc': train_acc_history,
                           'val_loss': val_loss_history, 'val_acc': val_acc_history})
    history.to_csv(f'{resFolder}/{model_name}_{stainFunc}_TrainValhistory_fold{foldcounter}.csv')



    print('-' * 30)
    print('START TESTING ...')
    print('-' * 30)
    model.load_state_dict(torch.load(f"{resFolder}/{model_name}_{stainFunc}_MILbestModel_fold{foldcounter}.pt"))
    model = model.to(device)
    testLoader = dls.test_dl(test_data)

    probsList, attention_scores  = test_model(model, testLoader)

    target_labelDict = {'<30y': 0, '30-50y': 1, '>50y': 2}
    probs = {}
    for key in list(target_labelDict.keys()): # <30y 30-50y >50y
        probs[key] = []
        for item in probsList:
            probs[key].append(item[int(target_labelDict[key])])


    probs = pd.DataFrame.from_dict(probs) # 0.2, 0.5, 0.3
    test_data = test_data.rename(columns = {'age_group': 'yTrueLabel'}) # cls->yTrueLabel
    test_data['yTrue'] = [target_labelDict[i] for i in test_data['yTrueLabel']] # 0,1,2
    testResults = pd.concat([test_data, probs], axis = 1)  # >50y, 2, 0.1, 0.3, 0.6                   
    
    for key in target_labelDict.keys():
        yTrueList = testResults['yTrue']
        yProbList = testResults[key]
        fpr, tpr, thresholds = metrics.roc_curve(yTrueList, yProbList, pos_label = target_labelDict[key])
        # print(fpr, tpr, thresholds)
        print(np.round(metrics.auc(fpr, tpr), 3)) 

    foldcounter += 1