# %%
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import models
import cv2 as cv
from tqdm import tqdm
from pydub import AudioSegment

# ========== CONFIG ==========
DEBUG_SMALL_MELS = False
N_MELS = 3401 if not DEBUG_SMALL_MELS else 256
N_FFT = 16384 if not DEBUG_SMALL_MELS else 2048
HOP_SEC = 0.015
N_FRAMES = 200
POWER_SPECT = 2
FMIN, FMAX = 300, 3400


BANDS_MEL = [
    (300, 627),
    (628, 1060),
    (1061, 1633),
    (1634, 2393),
    (2394, 3400)
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

EPS = 1e-9

# %%
# ========== SECTION 1: Spectral feature extraction  ==========
def compute_mel_spectrogram(y, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=None, fmin=FMIN, fmax=FMAX, power=POWER_SPECT):
    if hop_length is None:
        hop_length = int(sr * HOP_SEC)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                       hop_length=hop_length, fmin=fmin, fmax=fmax, power=power)
    return S, hop_length


def mel_bin_freqs(n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """
    Return an array of mel-bin *centers* in MEL units (not Hz).
    This intentionally returns MEL-scale values because the band limits
    provided (BANDS_MEL) are in mel units.
    """
    mel_min = librosa.hz_to_mel(fmin)
    mel_max = librosa.hz_to_mel(fmax)
    return np.linspace(mel_min, mel_max, n_mels)


def band_bin_indices(mel_bins, band):
    """Find indices of mel_bins that fall inside (inclusive) the provided band (in mel units)."""
    fmin_mel, fmax_mel = band
    idx = np.where((mel_bins >= fmin_mel) & (mel_bins <= fmax_mel))[0]
    if len(idx) == 0:
        
        idx = np.array([np.argmin(np.abs(mel_bins - fmin_mel)), np.argmin(np.abs(mel_bins - fmax_mel))])
    return idx


def compute_SC_SBW_SBE_from_mel(S, sr, mel_bins=None, bands=BANDS_MEL):
    """
    S: (n_mels, n_frames) mel spectrogram (power)
    mel_bins: mel-scale bin centers (MEL units), length n_mels

    Returns dict of arrays: SC, SBW, SBE each shape (n_bands, n_frames)

    Corrections applied:
    - SBW = variance as per tutor (no sqrt)
    - SBE = band-energy / total-energy (fraction)
    """
    n_mels = S.shape[0]
    if mel_bins is None:
        mel_bins = mel_bin_freqs(n_mels)
    n_frames = S.shape[1]
    n_bands = len(bands)
    SC = np.zeros((n_bands, n_frames), dtype=np.float32)
    SBW = np.zeros((n_bands, n_frames), dtype=np.float32)
    SBE = np.zeros((n_bands, n_frames), dtype=np.float32)

    total_energy = np.sum(S, axis=0) + EPS  

    for bi, band in enumerate(bands):
        idx = band_bin_indices(mel_bins, band)
        freqs = mel_bins[idx] 
        S_band = S[idx, :]  
        energy = np.sum(S_band, axis=0) + EPS  

        # Spectral centroid 
        num = np.sum((freqs[:, None] * S_band), axis=0)
        sc = num / energy

        # Variance about the centroid 
        diff_sq = (freqs[:, None] - sc[None, :]) ** 2
        var = np.sum(diff_sq * S_band, axis=0) / energy
        sbw = var  

        # Spectral band energy as fraction of total energy
        sbe = energy / total_energy

        SC[bi, :] = sc
        SBW[bi, :] = sbw
        SBE[bi, :] = sbe

    return {"SC": SC, "SBW": SBW, "SBE": SBE}


def truncate_or_pad_spectrogram(S, n_frames=N_FRAMES):
    n_mels, n = S.shape
    if n >= n_frames:
        return S[:, :n_frames]
    else:
        padded = np.zeros((n_mels, n_frames), dtype=S.dtype)
        padded[:, :n] = S
        return padded


def extract_spectral_feature_vector_from_file(wav_path, feature_type="SC", n_frames=N_FRAMES):
    y, sr = librosa.load(wav_path, sr=None)
    S, hop = compute_mel_spectrogram(y, sr)
    S = truncate_or_pad_spectrogram(S, n_frames=n_frames)
    mel_bins = mel_bin_freqs(n_mels=S.shape[0], fmin=FMIN, fmax=FMAX)
    feats = compute_SC_SBW_SBE_from_mel(S, sr, mel_bins=mel_bins)
    M = feats[feature_type]  
    # stack the columns: column-major flatten (each column stacked)
    vec = M.flatten(order='F')  # length n_bands * n_frames
    return vec


def build_features_for_dataset(data_dir, out_npz_path, feature_type="SC", n_frames=N_FRAMES):
    X, y = [], []
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print("Found labels:", labels)
    for label_idx, label in enumerate(labels):
        folder = os.path.join(data_dir, label)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(folder, fname)
            try:
                vec = extract_spectral_feature_vector_from_file(fpath, feature_type=feature_type, n_frames=n_frames)
                X.append(vec)
                y.append(label_idx)
            except Exception as e:
                print("Skipping", fpath, ":", e)
    X = np.stack(X) if len(X) > 0 else np.empty((0, len(BANDS_MEL) * n_frames))
    y = np.array(y, dtype=np.int64)
    np.savez_compressed(out_npz_path, X=X, y=y, labels=np.array(labels))
    print("Saved", out_npz_path, "X.shape=", X.shape, "y.shape=", y.shape)

# %%
# ========== SECTION 2: SVM classification  ==========
def train_and_eval_svm(train_npz, test_npz, C=0.1, kernel="linear", model_out="svm_model.joblib"):
    tr = np.load(train_npz, allow_pickle=True)
    te = np.load(test_npz, allow_pickle=True)
    X_train, y_train = tr["X"], tr["y"]
    X_test, y_test = te["X"], te["y"]
    labels = tr["labels"].tolist()
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = SVC(C=C, kernel=kernel)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("SVM accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    joblib.dump({"model": clf, "scaler": scaler, "labels": labels}, model_out)
    print("Saved SVM model to", model_out)
    return acc, cm, labels

# %%
# ========== SECTION 3.1: 1D CNN on raw waveform  ==========
class EmotionSpeechDataset(Dataset):
    def __init__(self, dataset_dir, n_frames=N_FRAMES, hop_length_in_sec=HOP_SEC, mean=1000.0, std=100.0):
        self.emotions = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        self.paths = [os.path.join(dataset_dir, e, p)
                      for e in self.emotions for p in sorted(os.listdir(os.path.join(dataset_dir, e))) if p.lower().endswith('.wav')]
        self.lengths = [len([p for p in os.listdir(os.path.join(dataset_dir, e)) if p.lower().endswith('.wav')]) for e in self.emotions]
        self.n_frames = n_frames
        self.hop_length_in_sec = hop_length_in_sec
        self.mean = mean
        self.std = std

    def index2label(self, index):
        num = 0
        for i in range(len(self.lengths)):
            num += self.lengths[i]
            if index < num:
                return i
        return len(self.lengths) - 1

    def normalise_mean_std(self, signal):
        signal = (signal - self.mean) / self.std
        return signal

    def index2signal(self, index):
        speech = AudioSegment.from_wav(self.paths[index])
        samples = np.array(speech.get_array_of_samples(), dtype=np.float32)
        sampling_rate = speech.frame_rate
        hop_length = int(sampling_rate * self.hop_length_in_sec)
        n_samples = (self.n_frames + 1) * hop_length
        signal = np.zeros(n_samples, np.float32)
        signal[:min(n_samples, len(samples))] = samples[:min(n_samples, len(samples))]
        signal = self.normalise_mean_std(signal)
        return signal

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        label = self.index2label(index)
        signal = self.index2signal(index)
        # return (1, time) shape tensor and label
        return torch.from_numpy(signal).unsqueeze(0).float(), label

class Simple1DCNN(nn.Module):
    def __init__(self, n_classes, drop1=0.1, drop2=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop1),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def train_1d_cnn(train_dir, test_dir, epochs=30, batch_size=32, lr=1e-3):
    train_ds = EmotionSpeechDataset(train_dir)
    test_ds = EmotionSpeechDataset(test_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model = Simple1DCNN(n_classes=len(train_ds.emotions)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x,y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        epoch_loss = running / len(train_ds)
        train_losses.append(epoch_loss)
        # Evaluate
        model.eval()
        ys, ypred = [], []
        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(DEVICE)
                out = model(x)
                preds = out.argmax(dim=1).cpu().numpy()
                ys.extend(y.numpy())
                ypred.extend(preds)
        acc = accuracy_score(ys, ypred)
        print(f"1D CNN Epoch {ep}/{epochs} train_loss={epoch_loss:.4f} test_acc={acc:.4f}")
    cm = confusion_matrix(ys, ypred)
    # Save model
    torch.save(model.state_dict(), "1d_cnn_model.pth")
    print("Saved 1D CNN: test_acc=", acc)
    return train_losses, acc, cm, train_ds.emotions

# %%
# ========== SECTION 3.2: 2D CNN on spectrogram images ==========

def spectrogram_to_image(X, filename):
    # X is magnitude/power spectrogram. Convert to dB and 0-255.
    X_db = librosa.power_to_db(X, ref=np.max)
    X_min, X_max = X_db.min(), X_db.max()
    X_scaled = (X_db - X_min) / (X_max - X_min + 1e-9) * 255.0
    X_uint8 = X_scaled.astype(np.uint8)
    # Save grayscale
    cv.imwrite(filename, X_uint8)


def save_all_spectrogram_images(data_dir, out_dir, n_mels=N_MELS, n_fft=N_FFT, power=1):
    # power=1 as assignment requires for 2D CNN part
    os.makedirs(out_dir, exist_ok=True)
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for label in labels:
        src_dir = os.path.join(data_dir, label)
        dst_dir = os.path.join(out_dir, label)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(src_dir, fname)
            y, sr = librosa.load(path, sr=None)
            hop_length = int(sr * HOP_SEC)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                               hop_length=hop_length, fmin=FMIN, fmax=FMAX, power=power)
            # Save as image 
            out_file = os.path.join(dst_dir, fname.replace(".wav", ".png"))
            spectrogram_to_image(S, out_file)
    print("Saved spectrogram images to", out_dir)

class Simple2DCNN(nn.Module):
    def __init__(self, n_classes, drop1=0.1, drop2=0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(drop1),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(drop2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Simple SpecAugment implementation 
class SpecAugment:
    def __init__(self, time_mask_param=20, freq_mask_param=10, num_masks=1):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks

    def __call__(self, x):
        # x is Tensor (C, H, W) where H is freq, W is time
        if not torch.is_tensor(x):
            return x
        x = x.clone()
        _, H, W = x.shape
        for _ in range(self.num_masks):
            # freq mask
            f = np.random.randint(0, min(self.freq_mask_param, H))
            f0 = np.random.randint(0, max(1, H - f))
            x[:, f0:f0+f, :] = 0
            # time mask
            t = np.random.randint(0, min(self.time_mask_param, W))
            t0 = np.random.randint(0, max(1, W - t))
            x[:, :, t0:t0+t] = 0
        return x


def train_2d_cnn(train_img_dir, test_img_dir, epochs=30, batch_size=32, lr=1e-3, augment=False):
    
    extra_transforms = []
    if augment:
        
        spec_aug = SpecAugment(time_mask_param=20, freq_mask_param=10, num_masks=1)
    else:
        spec_aug = None

    if spec_aug is None:
        transform = transforms.Compose([
            transforms.Resize((150, 50)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((150, 50)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            spec_aug,
            transforms.Normalize((0.5,), (0.5,))
        ])

    train_ds = ImageFolder(train_img_dir, transform=transform)
    test_ds = ImageFolder(test_img_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model = Simple2DCNN(n_classes=len(train_ds.classes)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x,y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        epoch_loss = running / len(train_ds)
        train_losses.append(epoch_loss)
        # eval
        model.eval()
        ys, ypred = [], []
        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(DEVICE)
                out = model(x)
                preds = out.argmax(dim=1).cpu().numpy()
                ys.extend(y.numpy())
                ypred.extend(preds)
        acc = accuracy_score(ys, ypred)
        print(f"2D CNN Epoch {ep}/{epochs} train_loss={epoch_loss:.4f} test_acc={acc:.4f}")
    cm = confusion_matrix(ys, ypred)
    torch.save(model.state_dict(), "2d_cnn_model.pth")
    print("Saved 2D CNN; final acc=", acc)
    return train_losses, acc, cm, train_ds.classes

# %%
# ========== UTIL: plotting and helpers ==========
def plot_learning_curve(losses, title, out_file):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_file)
    plt.close()
    print("Saved plot:", out_file)


def compute_and_print_metrics(y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    if labels is not None:
        print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    else:
        print(classification_report(y_true, y_pred, zero_division=0))
    return acc, cm


def print_dataset_stats(data_dir):
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print("Dataset:", data_dir)
    for label in labels:
        files = [f for f in os.listdir(os.path.join(data_dir, label)) if f.lower().endswith('.wav')]
        print(f"  {label}: {len(files)} files")



def answer_q2a_placeholder():
    
    print("Q2.a: Please add your analysis here (e.g., class-wise performance, explanation of features used, etc.)")


def answer_q2e_placeholder():
    print("Q2.e: Please add your answer here (e.g., discussion of augmentation, invariances, and why flips are not suitable for spectrograms).")

# %%
# ========== MAIN USAGE GUIDE ==========
if __name__ == "__main__":
    # Replace these paths with your dataset paths
    train_audio_dir = "/kaggle/input/es1102/EmotionSpeech/Train"
    test_audio_dir  = "/kaggle/input/es1102/EmotionSpeech/Test"

    
    print("NOTE: For debugging set DEBUG_SMALL_MELS=True to reduce memory and compute.")

    # 1) Extract spectral features and save
    for feature in ["SC", "SBW", "SBE"]:
        out_train = f"train_{feature}.npz"
        out_test  = f"test_{feature}.npz"
        print("Building features:", feature)
        build_features_for_dataset(train_audio_dir, out_train, feature_type=feature, n_frames=N_FRAMES)
        build_features_for_dataset(test_audio_dir, out_test, feature_type=feature, n_frames=N_FRAMES)

    # 2) Train SVM for each feature type and print confusion matrix
    for feature in ["SC", "SBW", "SBE"]:
        tr = f"train_{feature}.npz"
        te = f"test_{feature}.npz"
        print("="*30)
        print("SVM for feature:", feature)
        acc, cm, labels = train_and_eval_svm(tr, te, C=0.1, kernel="linear", model_out=f"svm_{feature}.joblib")
        print("Accuracy:", acc)
        print("Confusion matrix:\n", cm)

    # 3.1) Train a 1D CNN on raw waveforms
    print("="*30)
    print("Training 1D CNN (this may take a while)...")
    losses_1d, acc_1d, cm_1d, labels_1d = train_1d_cnn(train_audio_dir, test_audio_dir, epochs=30, batch_size=16)
    plot_learning_curve(losses_1d, "1D CNN Training Loss", "1dcnn_loss.png")
    print("1D CNN accuracy:", acc_1d)
    print("1D CNN confusion matrix:\n", cm_1d)

    # 3.2) Prepare spectrogram images for 2D CNN (power=1 as assignment)
    print("Saving spectrogram images for 2D CNN (power=1)...")
    save_all_spectrogram_images(train_audio_dir, "SpectrogramImages/Train", n_mels=N_MELS, n_fft=N_FFT, power=1)
    save_all_spectrogram_images(test_audio_dir,  "SpectrogramImages/Test",  n_mels=N_MELS, n_fft=N_FFT, power=1)

    # Train 2D CNN (no flips; use augment=True to apply SpecAugment (time/freq masks) if desired)
    print("Training 2D CNN...")
    losses_2d, acc_2d, cm_2d, labels_2d = train_2d_cnn("SpectrogramImages/Train", "SpectrogramImages/Test", epochs=30, batch_size=16, augment=False)
    plot_learning_curve(losses_2d, "2D CNN Training Loss", "2dcnn_loss.png")
    print("2D CNN accuracy:", acc_2d)
    print("2D CNN confusion matrix:\n", cm_2d)

    print("All done. Models and artifacts saved in the working directory.")



