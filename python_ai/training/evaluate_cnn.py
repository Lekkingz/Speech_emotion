"""Evaluate a trained CNN model and produce metrics and plots.

Produces: Accuracy, Precision, Recall, F1, Specificity, Confusion Matrix (image),
ROC curves, Loss/Accuracy curves saved under `models/trained/`.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

import joblib
import librosa
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'models' / 'trained'
DATASET_PATH = ROOT / 'datasets' / 'raw' / 'Audio_Speech_Actors_01-24'


def load_config():
    cfg_path = MODEL_DIR / 'cnn_config.json'
    if not cfg_path.exists():
        raise FileNotFoundError('cnn_config.json not found in models/trained')
    return json.loads(cfg_path.read_text())


def extract_mfcc_from_path(path, n_mfcc, max_pad_len):
    y, sr = librosa.load(str(path), sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc


def load_dataset(cfg):
    X = []
    y = []
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }
    for root, _, files in __import__('os').walk(DATASET_PATH):
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue
            try:
                code = fname.split('-')[2]
            except Exception:
                continue
            label = emotion_map.get(code, None)
            if label is None:
                continue
            file_path = Path(root) / fname
            mfcc = extract_mfcc_from_path(file_path, cfg['n_mfcc'], cfg['max_pad_len'])
            X.append(mfcc)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y


def plot_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    cfg = load_config()
    print('Loaded config:', cfg)

    model_path = MODEL_DIR / 'cnn_emotion_model.h5'
    if not model_path.exists():
        raise FileNotFoundError('Trained model not found at models/trained/cnn_emotion_model.h5')

    model = load_model(str(model_path))
    le = joblib.load(MODEL_DIR / 'cnn_label_encoder.pkl')

    X, y = load_dataset(cfg)
    X = X[..., np.newaxis]

    y_enc = le.transform(y)
    y_cat = np.eye(len(le.classes_))[y_enc]

    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_enc, y_pred)
    prec = precision_score(y_enc, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_enc, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_enc, y_pred, average='macro', zero_division=0)

    print(f'Accuracy: {acc:.4f}')
    print(f'Precision (macro): {prec:.4f}')
    print(f'Recall (macro): {rec:.4f}')
    print(f'F1 (macro): {f1:.4f}')

    # specificity per class from confusion matrix
    cm = confusion_matrix(y_enc, y_pred)
    tn = []
    fp = []
    fn = []
    tp = []
    for i in range(cm.shape[0]):
        tp_i = cm[i, i]
        fp_i = cm[:, i].sum() - tp_i
        fn_i = cm[i, :].sum() - tp_i
        tn_i = cm.sum() - (tp_i + fp_i + fn_i)
        tp.append(tp_i)
        fp.append(fp_i)
        fn.append(fn_i)
        tn.append(tn_i)
    specificity = [tn[i] / (tn[i] + fp[i]) if (tn[i] + fp[i]) > 0 else 0.0 for i in range(len(tp))]
    for cls, spec in zip(le.classes_, specificity):
        print(f'Specificity {cls}: {spec:.4f}')

    # save confusion matrix plot
    plot_confusion_matrix(cm, le.classes_, MODEL_DIR / 'cnn_confusion_matrix.png')

    # ROC curves (one-vs-rest)
    y_bin = label_binarize(y_enc, classes=list(range(len(le.classes_))))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(le.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), preds.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'], label=f'micro-average ROC (AUC = {roc_auc["micro"]:.2f})')
    for i in range(len(le.classes_)):
        plt.plot(fpr[i], tpr[i], lw=1, label=f'{le.classes_[i]} (AUC = {roc_auc[i]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(MODEL_DIR / 'cnn_roc_curves.png')
    plt.close()

    # Loss/Accuracy curves
    hist_path = MODEL_DIR / 'cnn_history.npy'
    if hist_path.exists():
        hist = np.load(hist_path, allow_pickle=True).item()
        plt.figure()
        plt.plot(hist.get('loss', []), label='loss')
        plt.plot(hist.get('val_loss', []), label='val_loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig(MODEL_DIR / 'cnn_loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(hist.get('accuracy', []), label='accuracy')
        plt.plot(hist.get('val_accuracy', []), label='val_accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig(MODEL_DIR / 'cnn_accuracy_curve.png')
        plt.close()

    print('Evaluation artifacts saved to', MODEL_DIR)


if __name__ == '__main__':
    main()
