import h5py
import random
import pandas as pd
from pathlib import Path
from typing import Tuple, Any
from typing import Optional
from fastai.vision.all import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from typing import List

# cross validation
# train test -> testSplit.csv
# train val -> trainValSplit.csv


            

def add_ageGroup(df):
    df['age_group'] = "30-50y"
    df.loc[df['age']<30, 'age_group'] = '<30y'
    df.loc[df['age']>50, 'age_group'] = '>50y'
    return df



def merge_feature_clinic(rootdir, clinic):
    for wsiname in os.listdir(rootdir):
        folder = os.path.join(rootdir, wsiname)
        fea_pt = glob.glob(f"{folder}/*patch.csv")
        
        if len(fea_pt)>0:
            new_pt = fea_pt[0].replace("patch.csv", "patch_merge.csv")
           
            if not os.path.exists(new_pt):

                fea_df=pd.read_csv(fea_pt[0])
                
                clinic_df = pd.read_csv(clinic)
                
                new_df = pd.merge(fea_df, clinic_df[['wsi_id', 'age']], on='wsi_id')
                new_df = add_ageGroup(new_df)
                new_df.to_csv(new_pt)
                
                print(f"{new_pt} saved!")
                

def get_patient_label(clinic_df, rootdir, model_name):

    patientsList = list(clinic_df['patient_id'])
    validPatients = []
    for i in patientsList:
        if len(glob.glob(f"{rootdir}/{i}*/*{model_name}*.h5")) >0:
            validPatients.append(i)
    df = clinic_df[clinic_df['patient_id'].isin(validPatients)]
    patientID = np.array(list(df['patient_id']))
    
    yTrueLabel = df['age_group']
    yTrueLabel = np.array(yTrueLabel)
    
    le = LabelEncoder()
    yTrue = le.fit_transform(yTrueLabel)  
    yTrue = np.array(yTrue)
    
    return patientID, yTrue, yTrueLabel




def feaBag_pt(patient_id, rootdir, model_name, stainFunc):
    feature_list = glob.glob(f"{rootdir}/{patient_id}*/*{model_name}*{stainFunc}.h5")
    index = random.randint(0, len(feature_list)-1)
    bag_pt = feature_list[index]
    return bag_pt




def split_data(clinic_df, rootdir, stainFunc, patientID, yTrue, train_index, test_index, fold_pt):
    
    print('preparing train/test patients, 0.9/0.1')
    train_patients = patientID[train_index]
    train_data = clinic_df[clinic_df['patient_id'].isin(train_patients)]
    train_data.reset_index(inplace=True, drop=True)
    
    test_patients = patientID[test_index]
    test_data = clinic_df[clinic_df['patient_id'].isin(test_patients)]
    test_data.reset_index(inplace=True, drop=True)
    
    print('preparing train/val patients, 0.9/0.1')
    val_data = train_data.groupby('age_group', group_keys=False).apply(lambda x: x.sample(frac=0.1))
    
    print('adding is_valid and feaBag_pt columns...')
    train_data = train_data.copy()
    train_data['is_valid'] = train_data['patient_id'].isin(list(val_data['patient_id']))

    model_name = os.path.basename(fold_pt).split('_')[0]
    
    feaBag_pts = [feaBag_pt(patient_id, rootdir, model_name, stainFunc) for patient_id in list(train_data['patient_id'])]
    train_data = train_data.copy()
    train_data['feaBag_pt'] = [Path(i) for i in feaBag_pts]
    train_data.to_csv(fold_pt, index = False)
    print(f'{fold_pt} saved!')
    
    feaBag_pts = [feaBag_pt(patient_id, rootdir, model_name, stainFunc) for patient_id in list(test_data['patient_id'])]
    test_data = test_data.copy()
    test_data['feaBag_pt'] = [Path(i) for i in feaBag_pts]
    test_data.to_csv(fold_pt.replace('train', 'test'), index = False)
    print(f"{fold_pt.replace('train', 'test')} saved!")
    
    return train_data, val_data, test_data




# given a bag features, make consistent bag size
def to_fixed_size_bag(bag: torch.Tensor, bag_size: int = 512) -> Tuple[torch.Tensor, int]:
    
    # get up to bag_size elements
    bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
    bag_samples = bag[bag_idxs]
    
    # zero-pad if we don't have enough samples
    zero_padded = torch.cat((bag_samples,
                              torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1])))
    return zero_padded, min(bag_size, len(bag))






class MILBagTransform(Transform):
        
    def __init__(self, valid_files, max_bag_size = 512, cls=['epithelium']):
        self.max_bag_size = max_bag_size
        self.cls = cls
        valid_files = [Path(file) for file in valid_files]
        self.valid = {f: self._draw(f) for f in valid_files}
        

    def encodes(self, f: Path):# -> Tuple[torch.Tensor, int]:
        return self.valid.get(f, self._draw(f))

    
    def _draw(self, f: Path) -> Tuple[torch.Tensor, int]: 
        csv_pt =  os.path.join(f.parent, f.stem.split('_bagFeature_')[0]+'_patch.csv')
        df = pd.read_csv(csv_pt) 
       
        with h5py.File(f, "r") as file:
            bag = np.array(file["embeddings"])
            img_id = np.array(file["patch_id"])
        img_id = [i.decode("utf-8") for i in img_id]
                
        bag_df = pd.DataFrame(bag)
        bag_df.index = img_id

        valid_ids = []
        for typei in self.cls:
            valid_id = list(df['patch_id'][df['cls'] == typei])
            random.shuffle(valid_id)
            valid_ids.extend(valid_id[:self.max_bag_size])
            
        bag_df = bag_df.loc[valid_ids, :]
        bag_df = np.squeeze(np.array(bag_df))
        bag_df = torch.from_numpy(bag_df)
        
        return to_fixed_size_bag(bag_df, bag_size = self.max_bag_size * len(self.cls))



def get_key_from_value(dataDict, value):
    for key in dataDict.keys():
        if dataDict[key] == value:
            return key
