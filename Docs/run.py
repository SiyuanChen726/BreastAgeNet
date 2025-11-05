srun -p interruptible_gpu --gres gpu:1 --constraint a100 --pty /bin/bash -l
cd /scratch/prj/cb_histology_data/Siyuan/Docker_test/singularity
singularity shell --writable --nv --bind /cephfs/volumes/hpc_data_prj/cb_normalbreast/5bd8c988-b2ab-4eb6-b25f-6e47c2c4b969/CAMBRIDGE:/app/project nbtclassifier_sandbox
source /opt/conda/etc/profile.d/conda.sh
conda activate nbtclassifier

# HistoQC
cd /app/HistoQC
python -m histoqc -c NBT -n 2 /app/project/WSIs/*.ndpi -o /app/project/QCs


# NBT-Classifier
cd /app/NBT-Classifier
python main.py \
  --wsi_folder /app/project/WSIs \
  --mask_folder /app/project/QCs \
  --output_folder /app/project/FEATUREs \
  --model_type TC_512 \
  --patch_size_microns 128 \
  --use_multithreading \
  --max_workers 32
exit


# BreastAgeNet
srun -p interruptible_gpu --gres gpu:1 --constraint a100 --pty /bin/bash -l
cd /scratch/prj/cb_histology_data/Siyuan/Docker_test/singularity
singularity shell --writable --nv --bind /cephfs/volumes/hpc_data_prj/cb_normalbreast/5bd8c988-b2ab-4eb6-b25f-6e47c2c4b969/CAMBRIDGE:/app/project breastagenet_sandbox

source /opt/conda/etc/profile.d/conda.sh
conda activate breastagenet
cd /app/project

nohup python run_breastagenet.py --start 28 --end 60 > breastagenet_28_60.log 2>&1 &
tail -f breastagenet_28_60.log

nohup python run_breastagenet.py --start 45 --end 60 > breastagenet_45_60.log 2>&1 &
tail -f breastagenet_45_60.log

nohup python run_breastagenet.py --start 0 --end 60 > breastagenet_0_60.log 2>&1 &

python
from utils.utils_model import *
def test_single_slide(wsi_path, patch_info, save_pt=None):
    # load UNI model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    UNI_model, transform = get_model(model_name='UNI', device=device)
    print(f"device: {device} \nmodel_name: UNI")
    # vectorise the WSI
    patch_df = pd.read_csv(patch_info)
    wsi_id = np.unique([i.split("_")[0] for i in patch_df["patch_id"]])[0]
    if save_pt is not None and os.path.exists(save_pt):
        df = pd.read_csv(save_pt)
        existing_wsi_ids = np.unique(df["wsi_id"])
        if wsi_id in existing_wsi_ids:
            print(f"BreastAgeNet has processed {wsi_id}!")
            return None
    bag_dataset = Dataset_fromWSI(patch_df, os.path.dirname(wsi_path), stainFunc='reinhard', transforms_eval=transform)
    h5_path = patch_info.replace("_TC_512_patch_all.csv", "_bagFeature_UNI_reinhard.h5") 
    print("extract UNI features...")
    extract_features(UNI_model, bag_dataset, batch_size=16, num_workers=2, device=device, fname=h5_path)
    # testloader
    valid_patches = list(patch_df['patch_id'][patch_df['TC_epi'] > 0.9])
    infer_df = pd.DataFrame({"h5df": [Path(h5_path)], "age_group": [0]})
    test_dblock = DataBlock(blocks = (TransformBlock, CategoryBlock),
                            get_x = ColReader('h5df'),
                            get_y = ColReader('age_group'),
                            item_tfms = MILBagTransform(infer_df.h5df, 250, 0.5, valid_patches))
    test_dls = test_dblock.dataloaders(infer_df, bs=1, shuffle=False)
    testloaders = test_dls.test_dl(infer_df, with_labels=True)
    # load BreastAgeNet
    print("loading BreastAgeNet...")
    ckpt_pt = "/app/BreastAgeNet/weights/epi0.9_UNI_250_MultiHeadAttention_full_best.pt"
    breastagenet = load_BreastAgeNet(ckpt_pt, embed_attn=True)
    # tissue ageing ranks predictions
    print("BreastAgeNet predicting...")
    predictions = test_model_iterations(breastagenet, testloaders, n_iteration = 10)
    predictions['wsi_id'] = [i.split("_")[0] for i in predictions["patch_id"]]
   # Save predictions with proper file existence check
    if save_pt is not None:
        if os.path.exists(save_pt):
            # Append without header if file exists
            predictions.to_csv(save_pt, mode='a', header=False, index=False)
        else:
            # Create new file with header if file doesn't exist
            predictions.to_csv(save_pt, index=False)
        print(f"Predictions saved to: {save_pt}")
    for col in ['branch_0', 'branch_1', 'branch_2']:
        predictions[col] = pd.to_numeric(predictions[col], errors='coerce')
    predictions_averaged = predictions.groupby('wsi_id')[['branch_0', 'branch_1', 'branch_2']].mean().reset_index()
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    for i in range(3):
        predictions_averaged[f'sigmoid_{i}'] = sigmoid(predictions_averaged[f'branch_{i}'])
        predictions_averaged[f'binary_{i}'] = (predictions_averaged[f'sigmoid_{i}'] >= 0.5).astype(int)
    predictions_averaged['final_prediction'] = predictions_averaged[['binary_0', 'binary_1', 'binary_2']].sum(axis=1)
    predictions_averaged = predictions_averaged.drop_duplicates("wsi_id")
    return predictions_averaged


save_pt = "/app/project/BreastAgeNet_predictions.csv"
WSIs = "/app/project/WSIs"
FEATURES = "/app/project/FEATUREs"
for i in os.listdir(WSIs):
    i = i.split('.ndpi')[0]
    wsi_path = f"{WSIs}/{i}.ndpi"
    patch_info = f"{FEATURES}/{i}/{i}_TC_512_patch_all.csv"
    predictions_averaged = test_single_slide(wsi_path, patch_info, save_pt=save_pt)
    print(predictions_averaged)
