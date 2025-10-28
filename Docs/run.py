srun -p interruptible_gpu --gres gpu:1 --constraint a100 --pty /bin/bash -l
cd /scratch/prj/cb_histology_data/Siyuan/Docker_test/singularity


# HistoQC
singularity shell --writable --nv --bind /scratch/prj/cb_normalbreast/WSIs_NEW2:/app/project nbtclassifier_sandbox
source /opt/conda/etc/profile.d/conda.sh
conda activate nbtclassifier
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
singularity shell --writable --nv --bind /scratch/prj/cb_normalbreast/WSIs_NEW2:/app/project breastagenet_sandbox
source /opt/conda/etc/profile.d/conda.sh
conda activate breastagenet
cd /app/BreastAgeNet
python
from utils.utils_model import test_single_slide
wsi_path = “/app/project/WSIs/23003346 IncN FPE-7-FPESec-1 - 2025-10-03 12.01.14.ndpi”
age_group = 0
patch_info = “/app/project/FEATUREs/23003346 IncN FPE-7-FPESec-1 - 2025-10-03 12.01.14/23003346 IncN FPE-7-FPESec-1 - 2025-10-03 12.01.14_TC_512_patch_all.csv”
test_single_slide(wsi_path, patch_info, age_group)
