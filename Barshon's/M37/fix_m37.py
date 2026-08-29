import json

cell_0 = """# M37 — Audio Spectrogram Transformer LoRA PEFT

**Model ID:** M37  
**Novelty Extension:** §4.1 — Foundation-model PEFT Adaptation (LoRA)  
**Contributor:** Barshon  
**Project:** OWMTL

## Objective
Apply **Low-Rank Adaptation (LoRA, Hu et al. 2022)** to the M2 CNN backbone's fully-connected layers, freezing the convolutional encoder and only training the LoRA decomposition matrices (rank r=8).

**Key Contribution:** Demonstrates that parameter-efficient fine-tuning achieves competitive performance while training only ~1% of total parameters, reducing compute and enabling efficient domain adaptation."""

cell_2 = """# ============================================================
# Section 1: Environment Setup & Dependencies
# ============================================================
import os, sys, re, time, json, math, glob, random, shutil, io, zipfile, tempfile
import base64, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support,
    classification_report
)

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
print(f'Device: {DEVICE} ({GPU_NAME})')
print(f'PyTorch: {torch.__version__} | Python: {sys.version.split()[0]}')"""

cell_4 = """# ============================================================
# Section 2: Configuration & Path Resolution
# ============================================================
# ---- Auto-detect Platform ----
if os.path.exists('/kaggle'):
    PLATFORM = 'Kaggle'
    BASE_DIR = '/kaggle/working'
elif os.path.exists('/content'):
    PLATFORM = 'Colab'
    BASE_DIR = '/content'
else:
    PLATFORM = 'Local'
    BASE_DIR = '.'
print(f'Platform: {PLATFORM}')

# ---- Google Drive Mount (Colab) ----
DRIVE_DIR = None
if PLATFORM == 'Colab':
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        DRIVE_DIR = '/content/drive/MyDrive/OWMTL/M37'
        os.makedirs(DRIVE_DIR, exist_ok=True)
    except Exception as e:
        print(f'Drive mount skipped ({e})')

# ---- ICBHI Dataset Path Resolution ----
POSSIBLE_ROOTS = [
    '/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files',
    '/kaggle/input/respiratory-sound-database/audio_and_txt_files',
    '/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/audio_and_txt_files',
    '/kaggle/input/icbhi-2017-respiratory-sound-database/audio_and_txt_files',
    '/content/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files',
    '/content/drive/MyDrive/respiratory-sound-database/audio_and_txt_files',
    '/content/drive/MyDrive/OWMTL/data/audio_and_txt_files',
    './data/audio_and_txt_files',
]
DATA_ROOT = next((p for p in POSSIBLE_ROOTS if os.path.exists(p)), None)
if DATA_ROOT is None and os.path.exists('/kaggle/input'):
    for root, dirs, files in os.walk('/kaggle/input'):
        if any(f.endswith('.wav') for f in files) and any(f.endswith('.txt') for f in files):
            DATA_ROOT = root
            print(f'Dynamic Kaggle resolution: {DATA_ROOT}')
            break

if DATA_ROOT and os.path.exists(DATA_ROOT):
    print(f'✅ ICBHI dataset verified: {DATA_ROOT}')
else:
    print(f'⚠️ DATA_ROOT not found — set DATA_ROOT manually')

# ---- M12/M2 Backbone Checkpoint Resolution ----
def resolve_checkpoint(candidates):
    return next((p for p in candidates if p and os.path.exists(p)), None)

M2_CKPT_PATH = resolve_checkpoint([
    '/content/M2_best_model.pth',
    '/kaggle/input/m2-checkpoint/best_model.pth',
    '/kaggle/input/owmtl-m2/best_model.pth',
    '/kaggle/input/m2-best-model/best_model.pth',
    '/content/drive/MyDrive/OWMTL/M2/best_model.pth',
    '../M2/best_model.pth',
    os.path.join(BASE_DIR, 'best_model.pth'),
])

CFG = {
    'model_id': 'M37',
    'model_name': 'Audio LoRA PEFT',
    'contributor': 'Barshon',
    'seed': SEED,
    # Shared Audio Parameters (Protocol §2)
    'sample_rate': 16000,
    'duration_s': 8.0,
    'n_mels': 128,
    'n_fft': 1024,
    'hop_length': 160,
    'win_length': 400,
    'f_min': 50,
    'f_max': 2000,
    'n_samples': int(16000 * 8.0),
    'n_frames': 1 + math.floor(128000 / 160),
    # Sound Event Classes (4)
    'sound_classes': ['Normal', 'Crackle', 'Wheeze', 'Both'],
    'num_classes': 4,
    # Training Hyperparameters
    'batch_size': 32,
    'num_epochs': 30,
    'lr': 0.0003,
    'weight_decay': 0.0001,
    'dropout': 0.4,
    'architecture': 'M2_LoRA_r8',
    'data_root': DATA_ROOT,
    'm2_ckpt_path': M2_CKPT_PATH,
    'ckpt_dir': os.path.join(BASE_DIR, 'checkpoints_M37'),
    'results_dir': os.path.join(BASE_DIR, 'results_M37'),
    'lora_rank': 8,
}

os.makedirs(CFG['ckpt_dir'], exist_ok=True)
os.makedirs(CFG['results_dir'], exist_ok=True)

print(f"\\n{'='*60}")
print(f'M37 CONFIGURATION — Audio LoRA PEFT')
print(f"{'='*60}")
for k, v in CFG.items():
    if 'path' in k or 'dir' in k:
        print(f'  {k}: {v}')
print(f"{'='*60}")"""

cell_6 = """# ============================================================
# Section 3: Real ICBHI Audio Loading & Patient-Independent Splitting
# ============================================================
try:
    import librosa
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'librosa'])
    import librosa

def extract_log_mel(wav_path, start, end, cfg):
    \"\"\"Extract normalized log-mel spectrogram from a respiratory cycle.\"\"\"
    sr, n_samples = cfg['sample_rate'], cfg['n_samples']
    try:
        audio, _ = librosa.load(wav_path, sr=sr, offset=start,
                                duration=max(end - start, 0.05), mono=True)
    except Exception as e:
        # A silent all-zero spectrogram here would be trained on and
        # scored as a real cycle. Fail instead of substituting
        # (Model_Training_Protocol.md section 1.2).
        raise RuntimeError(f"failed to load audio: {wav_path}") from e
    if len(audio) == 0:
        # Empty decode is a failed read, not a silent zero cycle.
        raise RuntimeError(f"empty audio decoded from audio: {wav_path}")
    # Pad or trim to fixed length
    if len(audio) < n_samples:
        audio = np.tile(audio, math.ceil(n_samples / len(audio)))[:n_samples]
    else:
        audio = audio[:n_samples]
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=cfg['n_mels'], n_fft=cfg['n_fft'],
        hop_length=cfg['hop_length'], win_length=cfg['win_length'],
        fmin=cfg['f_min'], fmax=cfg['f_max'], power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    T = log_mel.shape[1]
    if T < cfg['n_frames']:
        log_mel = np.pad(log_mel, ((0, 0), (0, cfg['n_frames'] - T)), mode='constant')
    else:
        log_mel = log_mel[:, :cfg['n_frames']]
    return log_mel[np.newaxis, :, :].astype(np.float32)

def parse_annotation_file(txt_path):
    \"\"\"Parse ICBHI annotation file into cycle list with labels.\"\"\"
    cycles = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4: continue
            try:
                start, end = float(parts[0]), float(parts[1])
                crackle, wheeze = int(parts[2]), int(parts[3])
            except ValueError: continue
            if end <= start: continue
            if crackle == 0 and wheeze == 0: label = 0
            elif crackle == 1 and wheeze == 0: label = 1
            elif crackle == 0 and wheeze == 1: label = 2
            else: label = 3
            cycles.append({'start': start, 'end': end, 'label': label})
    return cycles

def build_icbhi_splits(data_root, cfg):
    \"\"\"Load all ICBHI cycles and perform official patient-independent split.\"\"\"
    wav_paths = sorted(glob.glob(os.path.join(data_root, '*.wav')))
    if not wav_paths:
        raise FileNotFoundError(f'No .wav files under {data_root}')
    rows = []
    for wav_path in wav_paths:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        txt_path = os.path.join(data_root, stem + '.txt')
        if not os.path.exists(txt_path): continue
        try: pid = int(stem.split('_')[0])
        except (ValueError, IndexError): continue
        cycles = parse_annotation_file(txt_path)
        for c in cycles:
            rows.append({
                'wav_path': wav_path, 'stem': stem, 'patient_id': pid,
                'start': c['start'], 'end': c['end'], 'sound_label': c['label']
            })
    df = pd.DataFrame(rows)
    all_pids = sorted(df['patient_id'].unique())
    np.random.seed(SEED)
    np.random.shuffle(all_pids)
    n_train = int(len(all_pids) * 0.70)
    train_pids = set(all_pids[:n_train])
    test_pids = set(all_pids[n_train:])
    df_train = df[df['patient_id'].isin(train_pids)].reset_index(drop=True)
    df_test = df[df['patient_id'].isin(test_pids)].reset_index(drop=True)
    return df_train, df_test

class RealICBHI_SoundDataset(Dataset):
    \"\"\"ICBHI respiratory sound event dataset loading real .wav audio.\"\"\"
    def __init__(self, df, cfg):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec = extract_log_mel(row['wav_path'], row['start'], row['end'], self.cfg)
        return torch.from_numpy(spec), torch.tensor(row['sound_label'], dtype=torch.long)

df_train, df_test = build_icbhi_splits(CFG['data_root'], CFG)
print(f'Train set: {len(df_train)} cycles across {df_train["patient_id"].nunique()} patients')
print(f'Test set:  {len(df_test)} cycles across {df_test["patient_id"].nunique()} patients')

train_ds = RealICBHI_SoundDataset(df_train, CFG)
test_ds = RealICBHI_SoundDataset(df_test, CFG)
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False)

# Class weights (inverse frequency)
class_counts = df_train['sound_label'].value_counts().sort_index().values
class_weights = 1.0 / (class_counts.astype(np.float32) + 1e-6)
class_weights = class_weights / class_weights.sum()
CLASS_WEIGHTS = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f'Class counts:  {class_counts}')
print(f'Class weights: {class_weights.round(4)}')"""

cell_8 = """# ---- M2 CNN Backbone (same as M30/M2 architecture) ----
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool),
        )
    def forward(self, x): return self.block(x)

class M2_CNN(nn.Module):
    \"\"\"M2 CNN Backbone — 5-block architecture with 768-dim embedding.\"\"\"
    def __init__(self, num_classes=4, depth=5, base_width=48, dropout=0.4, fc_dim=128):
        super().__init__()
        channels = [base_width * (2 ** i) for i in range(depth)]  # [48, 96, 192, 384, 768]
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, num_classes),
        )
        self.embedding_dim = channels[-1]  # 768
    def get_embedding(self, x):
        return self.gap(self.encoder(x)).flatten(1)
    def forward(self, x):
        emb = self.get_embedding(x)
        emb = self.dropout(emb)
        return self.head(emb)

def smart_load_checkpoint(path, device):
    \"\"\"Load checkpoint handling both .pth and .zip formats.\"\"\"
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found: {path}')
    if zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path, 'r') as z:
                names = z.namelist()
                target = 'best_model.pth'
                if target not in names:
                    target = next((n for n in names if n.endswith('.pth')), None)
                if target:
                    with z.open(target) as f:
                        return torch.load(io.BytesIO(f.read()), map_location=device, weights_only=False)
        except Exception:
            pass
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return torch.load(path, map_location=device, weights_only=True)

# ---- LoRA Linear Layer ----
class LoRALinear(nn.Module):
    \"\"\"Low-Rank Adaptation (LoRA) wrapper for nn.Linear.
    Freezes the base weight and adds trainable A/B decomposition matrices.
    \"\"\"
    def __init__(self, in_features, out_features, r=8, alpha=16.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r
    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

class M37_LoRA_CNN(nn.Module):
    \"\"\"M2 backbone with LoRA-adapted FC head.
     Encoder is frozen; only LoRA matrices and classifier bias are trainable.\"\"\"
    def __init__(self, num_classes=4, depth=5, base_width=48, dropout=0.4, fc_dim=128, lora_r=8):
        super().__init__()
        channels = [base_width * (2 ** i) for i in range(depth)]
        blocks, in_ch = [], 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        
        # LoRA-adapted FC layers
        self.fc1 = LoRALinear(channels[-1], fc_dim, r=lora_r)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = LoRALinear(fc_dim, num_classes, r=lora_r)
        self.embedding_dim = channels[-1]
        
    def forward(self, x):
        emb = self.gap(self.encoder(x)).flatten(1)
        emb = self.dropout(emb)
        emb = self.relu(self.fc1(emb))
        return self.fc2(emb)

# Instantiate
model = M37_LoRA_CNN(num_classes=CFG['num_classes'], dropout=CFG['dropout'],
                     lora_r=CFG.get('lora_rank', 8)).to(DEVICE)

# Load M2 weights into frozen encoder
if CFG['m2_ckpt_path']:
    try:
        ckpt_m2 = smart_load_checkpoint(CFG['m2_ckpt_path'], DEVICE)
        sd = ckpt_m2.get('model_state', ckpt_m2.get('model_state_dict', ckpt_m2))
        # Load encoder weights (partial match)
        model_sd = model.state_dict()
        matched = 0
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                matched += 1
            elif k.startswith('encoder.') and k in model_sd:
                model_sd[k] = v
                matched += 1
        model.load_state_dict(model_sd, strict=False)
        print(f'✅ Loaded {matched} M2 weight tensors')
    except Exception as e:
        print(f'⚠️ M2 load: {e}')

# Freeze encoder
for name, param in model.named_parameters():
    if 'encoder' in name or 'gap' in name:
        param.requires_grad = False
    # LoRA A/B matrices and biases remain trainable

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total Params:     {total_params:,}')
print(f'Trainable (LoRA): {trainable_params:,} ({trainable_params/total_params:.1%})')"""

cell_12 = """# ============================================================
# Section 6: Comprehensive Evaluation & Visualizations
# ============================================================
# Load best checkpoint
ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state'])
best_ep = int(ckpt['epoch'])

loss, acc, f1_val, icbhi, targets, preds = eval_epoch(model, test_loader, criterion, DEVICE)
cm = confusion_matrix(targets, preds, labels=list(range(4)))
cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)
prec_macro = precision_score(targets, preds, average='macro', zero_division=0)
rec_macro = recall_score(targets, preds, average='macro', zero_division=0)

spec_per_class = []
for i in range(4):
    tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    spec_per_class.append(float(tn / (tn + fp + 1e-6)))
spec_macro = float(np.mean(spec_per_class))

prec_per = precision_score(targets, preds, average=None, zero_division=0, labels=list(range(4)))
rec_per = recall_score(targets, preds, average=None, zero_division=0, labels=list(range(4)))
f1_per = f1_score(targets, preds, average=None, zero_division=0, labels=list(range(4)))
support_per = [int(np.sum(np.array(targets) == i)) for i in range(4)]

# Model size
with io.BytesIO() as b:
    torch.save(model.state_dict(), b)
    model_size_mb = len(b.getvalue()) / (1024 * 1024)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Inference time
model.eval()
dummy = torch.randn(1, 1, CFG['n_mels'], CFG['n_frames']).to(DEVICE)
times_inf = []
with torch.no_grad():
    for _ in range(50):
        t0 = time.time()
        _ = model(dummy)
        times_inf.append((time.time() - t0) * 1000)
inf_ms = float(np.median(times_inf))

print(f'\\n{"="*60}')
print(f'M37 FINAL EVALUATION RESULTS')
print(f'{"="*60}')
print(f'  Best Epoch:       {best_ep}')
print(f'  Test Accuracy:    {acc:.4f}')
print(f'  Macro Precision:  {prec_macro:.4f}')
print(f'  Macro Recall:     {rec_macro:.4f}')
print(f'  Macro F1:         {f1_val:.4f}')
print(f'  Macro Specificity:{spec_macro:.4f}')
print(f'  ICBHI Score:      {icbhi:.4f}')
print(f'  Model Size:       {model_size_mb:.2f} MB')
print(f'  Total Params:     {total_params:,}')
print(f'  Trainable Params: {trainable_params:,}')
print(f'  Inference Time:   {inf_ms:.2f} ms/sample')
print(f'{"="*60}')

# ---- Visualization (§5 Required Plots) ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_list = [h['epoch'] for h in history]
train_l = [h['train_loss'] for h in history]
val_l = [h['val_loss'] for h in history]
val_a = [h['val_accuracy'] for h in history]
val_f1_list = [h['val_f1_macro'] for h in history]

# Plot 1: Loss curves
ax = axes[0, 0]
ax.plot(epochs_list, train_l, 'b-o', markersize=3, label='Train Loss')
ax.plot(epochs_list, val_l, 'r-s', markersize=3, label='Val Loss')
ax.axvline(best_ep, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_ep})')
ax.set_title('M37 — Loss Curves')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 2: Accuracy & ICBHI Score
ax = axes[0, 1]
val_icbhi_list = [h['val_icbhi_score'] for h in history]
ax.plot(epochs_list, val_a, 'g-o', markersize=3, label='Val Accuracy')
ax.plot(epochs_list, val_icbhi_list, 'm-s', markersize=3, label='Val ICBHI Score')
ax.axvline(best_ep, color='green', linestyle='--', alpha=0.7)
ax.set_title('M37 — Performance Curves')
ax.set_xlabel('Epoch'); ax.set_ylabel('Score')
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 3: Raw Confusion Matrix
ax = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CFG['sound_classes'], yticklabels=CFG['sound_classes'], ax=ax)
ax.set_title('Raw Confusion Matrix')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')

# Plot 4: Normalized Confusion Matrix
ax = axes[1, 1]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=CFG['sound_classes'], yticklabels=CFG['sound_classes'], ax=ax)
ax.set_title('Normalized Confusion Matrix')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')

plt.suptitle('M37 — M37 Evaluation', fontsize=14, y=1.01)
plt.tight_layout()
for d in sorted(set([CFG['results_dir'], BASE_DIR])):
    fig.savefig(os.path.join(d, 'm37_results.png'), dpi=150, bbox_inches='tight')
print('Saved: m37_results.png')
plt.show()
plt.close()"""

cell_14 = """# ============================================================
# Section 7: Exporting Protocol-Compliant Results JSON (§4 Schema)
# ============================================================
per_class_dict = {}
for i, cls_name in enumerate(CFG['sound_classes']):
    per_class_dict[cls_name] = {
        'precision': round(float(prec_per[i]), 4),
        'recall': round(float(rec_per[i]), 4),
        'f1': round(float(f1_per[i]), 4),
        'specificity': round(spec_per_class[i], 4),
        'support': support_per[i],
    }

results = {
    'meta': {
        'model_id': 'M37',
        'model_name': 'Audio LoRA PEFT Adaptation',
        'contributor': 'Barshon',
        'date_completed': datetime.datetime.now().strftime('%Y-%m-%d'),
        'is_augmented': False,
        'augmentation_method': 'none',
        'notes': 'Novelty Search §4.1 - Foundation Model PEFT Adaptation (LoRA)',
    },
    'config': {
        'sample_rate': CFG['sample_rate'],
        'n_mels': CFG['n_mels'],
        'batch_size': CFG['batch_size'],
        'num_epochs': CFG['num_epochs'],
        'lr': CFG['lr'],
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR',
        'architecture': CFG['architecture'],
        'seed': CFG['seed'],
    },
    'environment': {
        'platform': PLATFORM,
        'gpu_name': GPU_NAME,
        'pytorch_version': torch.__version__,
        'python_version': sys.version.split()[0],
    },
    'dataset_info': {
        'dataset': 'ICBHI_2017',
        'data_source': 'real_audio',
        'train_samples': int(len(df_train)),
        'test_samples': int(len(df_test)),
        'train_patients': int(df_train['patient_id'].nunique()),
        'test_patients': int(df_test['patient_id'].nunique()),
        'split_method': 'patient_independent_70_30',
    },
    'efficiency': {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'model_size_mb': round(float(model_size_mb), 2),
        'training_time_total_s': round(float(total_train_time), 2),
        'training_time_per_epoch_s_avg': round(float(total_train_time / max(CFG['num_epochs'], 1)), 2),
        'gpu_name': GPU_NAME,
        'inference_time_ms_per_sample': round(inf_ms, 2),
    },
    'best_epoch': {
        'epoch': int(best_ep),
        'primary_metric': 'icbhi_score',
        'primary_metric_value': round(float(icbhi), 4),
    },
    'best_metrics': {
        'accuracy': round(float(acc), 4),
        'precision_macro': round(float(prec_macro), 4),
        'recall_macro': round(float(rec_macro), 4),
        'f1_macro': round(float(f1_val), 4),
        'specificity_macro': round(spec_macro, 4),
        'icbhi_score': round(float(icbhi), 4),
        'per_class': per_class_dict,
        'confusion_matrix_raw': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.round(4).tolist(),
    },
    'ablation': {
        'ablation_group': 'peft_adaptation',
        'ablation_role': 'variant',
        'baseline_model_id': 'M2',
        'variable_changed': 'peft: LoRA rank r=8 on M2 FC layers (frozen encoder)',
        'variables_held_constant': [
            'loss_function: inverse_frequency_CrossEntropyLoss',
            'data_split: patient_independent_70_30',
            'seed: 42',
            'preprocessing: 128mel_16kHz_8s',
        ],
        'component_flags': {
            'has_sound_event_head': True,
            'has_disease_head': False,
            'has_cross_task_consistency': False,
            'has_cqkd_regularization': False,
            'has_openmax_rejection': False,
            'owl_stage': 0,
            'compression_clusters': None,
            'has_lora_adaptation': True,
        },
        'loss_weights': {
            'sound_event_weight': 1.0,
            'disease_weight': None,
            'consistency_weight': None,
        },
    },
    'training_history': history,
}

json_path = os.path.join(CFG['results_dir'], 'results_M37.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'✅ Saved: {json_path}')

# Also save to model folder if running locally
local_dir = os.path.join(BASE_DIR, "Barshon's", "M37")
if os.path.isdir(local_dir):
    local_json = os.path.join(local_dir, 'results_M37.json')
    with open(local_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'✅ Saved copy: {local_json}')"""


cell_15 = """## Section 8: Summary & Key Takeaways

**Model:** M37 — Audio LoRA PEFT
**Novelty Item:** §4.1 — Foundation-model PEFT Adaptation (LoRA)

**Key Results:**
- All metrics computed on **real ICBHI audio** with patient-independent splits
- Protocol-compliant `results_M37.json` with all 9 required blocks
- Best model checkpoint saved at `checkpoints_M37/best_model.pth`

**What This Means for the Novelty Claim:**
Tests whether Low-Rank Adaptation (training ~1% of parameters) achieves competitive performance to full fine-tuning, enabling efficient domain adaptation."""

cell_17 = """# ============================================================
# Section 8: Team Handoff & One-Click File Downloads (§11.D)
# ============================================================
from IPython.display import display, FileLink

print("=" * 60)
print("OFFICIAL PROTOCOL OUTPUTS READY FOR DOWNLOAD")
print("=" * 60)

protocol_files = sorted(
    glob.glob(os.path.join(CFG['ckpt_dir'], 'best_model.pth')) +
    glob.glob(os.path.join(CFG['results_dir'], 'results_M37.json')) +
    glob.glob(os.path.join(CFG['results_dir'], '*.png'))
)

for fpath in protocol_files:
    if os.path.exists(fpath):
        size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
        print(f"Ready: {os.path.basename(fpath):<25} ({size_mb} MB)")
        display(FileLink(fpath))
    else:
        print(f"Missing: {os.path.basename(fpath)}")

bundle_dir = os.path.join(BASE_DIR, 'protocol_bundle_M37')
if protocol_files:
    os.makedirs(bundle_dir, exist_ok=True)
    for fpath in protocol_files:
        if os.path.exists(fpath):
            shutil.copy2(fpath, os.path.join(bundle_dir, os.path.basename(fpath)))
    zip_path = shutil.make_archive(
        os.path.join(BASE_DIR, 'm37_handoff_bundle'), 'zip', bundle_dir)
    size_zip = round(os.path.getsize(zip_path) / (1024 * 1024), 2)
    print(f"\\nZIP bundle ({size_zip} MB):")
    display(FileLink('m37_handoff_bundle.zip'))

print("=" * 60)"""

def to_source(text):
    return [line + '\\n' for line in text.split('\\n')]

with open("M37_Audio_LoRA.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

nb['cells'][0]['source'] = to_source(cell_0)
nb['cells'][2]['source'] = to_source(cell_2)
nb['cells'][4]['source'] = to_source(cell_4)
nb['cells'][6]['source'] = to_source(cell_6)
nb['cells'][8]['source'] = to_source(cell_8)
nb['cells'][12]['source'] = to_source(cell_12)
nb['cells'][14]['source'] = to_source(cell_14)
nb['cells'][15]['source'] = to_source(cell_15)
nb['cells'][17]['source'] = to_source(cell_17)

# Make sure cell 10 stays as is.

with open("M37_Audio_LoRA.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
