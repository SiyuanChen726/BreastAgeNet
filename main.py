

from utils import create_data_split
from utils.core_utils import train

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



@hydra.main(version_base=None, config_path="/scratch/users/k21066795/prj_normal/BreastAgeNet_MIL/experiments/", config_name="config")
def run(config : DictConfig) -> None:
    seed_torch(config.seed)

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_test_acc = []

    for i in folds:
        train_dataset, val_dataset, test_dataset = create_data_split()
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


        
    









if __name__ == "__main__":
    run()  
    

print('It is a' + str(args.k) + 'fold cross validation for' + targetLabel + '!')
patientID = np.array(patientsList)
yTrue = np.array(yTrue)
yTrueLabel = np.array(yTrueLabel)

folds = args.k
kf = StratifiedKFold(n_splis=folds, random_state=args.seed, shuffle=True)
kf.get_n_splits(patientID, yTrue)

foldcounter = 1

for train_index, test_index in kf.split(patientID, yTrue):

    data = pd.read_csv(args.csvFile)
    test_patients = patientID[test_index]
    train_patients = patientID[train_index]
    train_data = data[data['patient_id'].isin(train_patients)]
    train_data.reset_index(inplace=True, drop=True)
    test_data = data[data['patient_id'].isin(test_patients)]
    test_data.reset_index(inplace = True, drop = True)

    val_data = train_data.groupby(args.label, group_keys=False).apply(lamda x: x.sample(frac=0.1))
    train_data['is_valid'] = train_data.PATIENT.isin(val_data['patient'])
    train_data['SlideAdr'] = [i.replace('BLOCKS_NORM_MACENKO', 'FEATURES') for i in train_data['SlideAdr']]
    train_data['SlideAdr'] = [Path(i + '.pt') for i in train_data['SlideAdr']]
    train_data.to_csv(os.path.join(args.split_dir, 'TrainValSplit.csv'), index = False)  

    test_data['SlideAdr'] = [i.replace('BLOCKS_NORM_MACENKO', 'FEATURES') for i in test_data['SlideAdr']]
    test_data['SlideAdr'] = [Path(i + '.pt') for i in test_data['SlideAdr']]
    test_data.to_csv(os.path.join(args.split_dir, 'TestSplit.csv'), index = False) 

    print('-' * 30)
    print("K FOLD VALIDATION STEP => {}".format(foldcounter))  
    print('-' * 30) 

    dblock = DataBlock(
        blocks = (TransformBlock, CategoryBlock),
        get_x = ColReader('SlideAdr'),
        get_y = ColReader(args.target_label),
        splitter = ColSplitter('is_valid'),
        item_tfms = MILBagTransform(train_data[train_data.is_valid].SlideAdr, 4096)
    )

    dls = dblock.dataloaders(train_data, bs = args.batch_size)
    weight = train_data[args.target_label].value_counts().sum() / train_data[args.target_label].value_counts()
    weight /= weight.sum()
    weight = torch.tensor(list(map(weight.get, dls.vocab)))
    criterion = CrossEntropyLossFlat(weight = weight.to(torch.float32))
    model = MILModel(1024, args.num_classes)
    model = model.to(device)
    criterion.to(device)
    optimizer = utils.get_optim(model, args, params=False)
    















    
    














