import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import pandas as pd
import os

from models import MfamaNet


MODEL_CONFIG = dict(
    num_classes=2, num_hiddens=96, dropout=0.3,
    signal_chans=6, scale_lst=[625, 1250, 2500, 5000, 6250],
    num_layers=[2, 4, 6, 8, 8],
    image_chans=3, num_blocks=[2, 4, 4, 6, 6],
    scale_kernels=[3, 5, 7, 11, 17], drop_path_rate=0.1,
    tre_type='token',
    use_mfa=False,
)

PARAMS_DIR = os.path.join(os.path.dirname(__file__), 'params')
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'samples')

IMAGE_TRANSFORM = T.Compose([
    T.Resize([224, 224], antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLASS_NAMES = {0: 'Healthy', 1: 'Patient'}


def load_model(task):
    """Load MfamaNet with task-specific weights (meander / spiral / circle)."""
    model = MfamaNet(**MODEL_CONFIG)
    weight_path = os.path.join(PARAMS_DIR, f'{task}.pth')
    state = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(image_path):
    """Load and preprocess a handwriting image."""
    img = Image.open(image_path).convert('RGB')
    return IMAGE_TRANSFORM(img).unsqueeze(0)  # [1, 3, 224, 224]


def preprocess_signal(signal_path, target_len=10000, trim_ratio=0.05):
    """Load and preprocess a handwriting sensor signal.

    Returns tensor of shape [1, target_len, 6].
    """
    sig = torch.as_tensor(
        pd.read_csv(signal_path).values, dtype=torch.float32)  # [L, 6]

    L = sig.shape[0]
    if trim_ratio > 0 and L > 50:
        trim_len = int(L * trim_ratio)
        if L - 2 * trim_len > 20:
            sig = sig[trim_len: L - trim_len]

    # Per-channel Z-score normalisation
    mean = sig.mean(dim=0, keepdim=True)
    std = sig.std(dim=0, keepdim=True).clamp(min=1e-8)
    sig = (sig - mean) / std

    # Interpolate to fixed length
    sig = sig.T.unsqueeze(0)                               # [1, C, L]
    sig = F.interpolate(sig, size=target_len, mode='linear',
                        align_corners=True)                # [1, C, target_len]
    sig = sig.squeeze(0).T                                 # [target_len, C]
    return sig.unsqueeze(0)                                # [1, target_len, C]


def predict(model, image_path, signal_path):
    """Run inference and return (predicted_class_id, logits)."""
    img = preprocess_image(image_path)
    sig = preprocess_signal(signal_path)

    with torch.no_grad():
        logits, _ = model(img, sig)          # [1, 2]
        pred = logits.argmax(dim=-1).item()
    return pred, logits.squeeze(0)


if __name__ == '__main__':
    for task in ['meander', 'spiral', 'circle']:
        print(f"{'='*50}")
        print(f"  Task: {task}")
        print(f"{'='*50}")
        model = load_model(task)

        for label in ['healthy', 'patient']:
            img_path = os.path.join(SAMPLES_DIR, task, f'{label}.jpg')
            sig_path = os.path.join(SAMPLES_DIR, task, f'{label}.csv')

            pred, logits = predict(model, img_path, sig_path)
            probs = F.softmax(logits, dim=-1)
            print(f"  {label:>8}:  pred={CLASS_NAMES[pred]:<8}  "
                  f"probs=[{probs[0]:.4f}, {probs[1]:.4f}]")
        print()
