# Multi-branch multiple-instance ordinal classification-based _BreastAgeNet_ identified deviated tissue ageing in high-risk breast tissues


_BreastAgeNet_ is a computational model designed to assess tissue aging in histologically normal breast tissue (NBT) from whole slide images (WSIs).

The framework leverages advanced deep learning methods, incorporating a multi-branch multiple-instance learning (MIL) architecture to capture subtle age-related alterations in breast tissue. 

<center>
  <img src="Docs/BreastAgeNet.png" width="60%">
</center>


It provides an ordinal classification of tissue aging into four distinct categories: <35 years, 35-45 years, 45-55 years, and >55 years.

<center>
  <img src="Docs/UR_NBT_ageing_prediction.png" width="80%">
</center>


This enables a more nuanced understanding of age-related changes in breast tissue by leveraging the attention mechanism across multiple branches

<center>
  <img src="Docs/BreastAgeNet_attention.png" width="80%">
</center>


Moreover, _BreastAgeNet_ generates attention heatmaps that reveal ageing-related heterogeneity across breast tissue, with this variability showing strong associations with manually annotated, age-related lobule types. This suggests the model's ability to pinpoint localized aging diversity, which can be important for understanding subtle variations in aging processes within the same breast.

<figure style="text-align:center;">
  <img src="Docs/attention_lobuletype_association.png" width="60%">
</figure>


With its substantiated ability to model aging trajectories in NBT, _BreastAgeNet_ has revealed deviations between expected (chronological) and observed (predicted) tissue aging in high-risk NBT from _gBRCA1/2_ mutation carriers or breast cancer patients. 

<center>
  <img src="Docs/HR_NBT_ageing_prediction.png" width="80%">
</center>


Taking it a step further, attention heatmaps can pinpoint tissue regions responsible for 'mismatched' tissue aging predictions. This approach opens the door to techniques like spatial transcriptomics, which could further elucidate molecular abnormalities at these sites—potentially identifying early indicators of cancer initiation.





## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [License](#license)
5. [Acknowledgments](#acknowledgments)


## Installation

To get started, clone the repository and install the required dependencies.

### Clone the repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt


### 5. Usage

Provide clear instructions on how to use your project, including code examples if necessary.

```markdown
## Usage

Once the environment is set up, you can start using the tool.

### Running the model

To train the model on your data, use the following command:

```bash
python train_model.py --data_dir /path/to/data --epochs 10


python visualize_results.py --model_path /path/to/model



