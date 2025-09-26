import os
import zipfile
import subprocess
from huggingface_hub import login, HfApi
from datasets import load_dataset

# Function to run shell commands
def run_command(cmd):
    try:
        subprocess.run(cmd, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}': {e}")
        raise

# Create data and datasets directories
data_dir = 'data'
datasets_dir = os.path.join(data_dir, 'datasets')
uncompressed_dir = os.path.join(data_dir, 'uncompressed')
os.makedirs(datasets_dir, exist_ok=True)
os.makedirs(uncompressed_dir, exist_ok=True)

# packages are installed via conda

# Hugging Face login (prompt for token if needed)
try:
    api = HfApi()
    user = api.whoami()
    print(f"Already logged in to Hugging Face as {user['name']}")
except:
    token = input("Enter your Hugging Face token: ")
    login(token)
    print("Logged in to Hugging Face")

# Download HF datasets if not present
pile_path = os.path.join(datasets_dir, 'ai_text_detection_pile')
if not os.path.exists(pile_path):
    print("Downloading AI Text Detection Pile...")
    pile_dataset = load_dataset("artem9k/ai-text-detection-pile")
    pile_dataset.save_to_disk(pile_path)
else:
    print("AI Text Detection Pile already present.")

hc3_path = os.path.join(datasets_dir, 'hc3')
if not os.path.exists(hc3_path):
    print("Downloading HC3...")
    hc3_dataset = load_dataset("Hello-SimpleAI/HC3", "all")
    hc3_dataset.save_to_disk(hc3_path)
else:
    print("HC3 already present.")

# Unzip existing zips in uncompressed_dir
zips_to_unzip = [
    ('AHAIRD_Dataset.zip', 'ah_aitd', 'AHAIRD_Dataset.xlsx'),  # Actual unzipped file
    ('sunilthite_Training_Essay_Data.zip', 'sunilthite', 'sunilthite_Training_Essay_Data.csv'),  # Actual unzipped file
    ('DAIGT_v2_train_v2_drcat_02.zip', 'daigt_v2', 'DAIGT_v2_train_v2_drcat_02.csv'),  # Actual unzipped file
    ('kaggleComp_train_essays.zip', 'llm_detect_competition', 'kaggleComp_train_essays.csv')  # Actual unzipped file
]

for zip_name, extract_subdir, main_file in zips_to_unzip:
    zip_path = os.path.join(uncompressed_dir, zip_name)
    extract_path = os.path.join(datasets_dir, extract_subdir)
    main_file_path = os.path.join(extract_path, main_file)
    
    if not os.path.exists(main_file_path):
        if os.path.exists(zip_path):
            print(f"Unzipping {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        else:
            print(f"{zip_name} not found in {uncompressed_dir}; please place it there.")
    else:
        print(f"{extract_subdir} already unzipped.")

# Verify all datasets are present
dataset_checks = [
    (pile_path, "AI Text Detection Pile folder"),
    (hc3_path, "HC3 folder"),
    (os.path.join(datasets_dir, 'sunilthite', 'sunilthite_Training_Essay_Data.csv'), "sunilthite CSV"),
    (os.path.join(datasets_dir, 'daigt_v2', 'DAIGT_v2_train_v2_drcat_02.csv'), "DAIGT V2 CSV"),
    (os.path.join(datasets_dir, 'llm_detect_competition', 'kaggleComp_train_essays.csv'), "Kaggle Competition train_essays.csv"),
    (os.path.join(datasets_dir, 'ah_aitd', 'AHAIRD_Dataset.xlsx'), "AH&AITD XLSX")
]

all_present = True
for path, desc in dataset_checks:
    if not os.path.exists(path):
        print(f"Missing: {desc} at {path}")
        all_present = False

if all_present:
    print("✅ All datasets are correctly downloaded and present.")
else:
    print("❌ Some datasets are missing; please check errors above.")

