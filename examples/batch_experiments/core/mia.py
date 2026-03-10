import numpy as np
import torch
import torch.nn as nn


def _collect_target_probs(model, loader, device, target_class, mode, max_samples=2000):
    feats = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            y_np = y.numpy()
            for i, label in enumerate(y_np):
                if mode == "target" and label == target_class:
                    feats.append(prob[i])
                elif mode == "remain" and label != target_class:
                    feats.append(prob[i])
                if len(feats) >= max_samples:
                    return np.asarray(feats)
    if not feats:
        return np.empty((0, 1))
    return np.asarray(feats)


def compute_fr(base_model, candidate_model, train_loader, test_loader, device, target_class):
    try:
        from sklearn import svm
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        print(f"[Warn] sklearn unavailable for Fr calculation: {e}")
        return 0.0

    # Member positives come from train target-class samples.
    pos = _collect_target_probs(base_model, train_loader, device, target_class, "target", max_samples=1500)
    # Unseen negatives come from test target-class samples.
    neg = _collect_target_probs(base_model, test_loader, device, target_class, "target", max_samples=max(1500, len(pos)))

    if len(pos) == 0 or len(neg) == 0:
        return 0.0

    n = min(len(pos), len(neg))
    x_train = np.concatenate([pos[:n], neg[:n]], axis=0)
    y_train = np.concatenate([np.ones(n), np.zeros(n)], axis=0)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    clf = svm.SVC(kernel="linear", probability=True, random_state=42)
    clf.fit(x_train, y_train)

    target_feats = _collect_target_probs(candidate_model, train_loader, device, target_class, "target", max_samples=2000)
    if len(target_feats) == 0:
        return 0.0

    target_feats = scaler.transform(target_feats)
    target_pred = clf.predict(target_feats)
    target_acc = np.mean(target_pred == 1)
    fr = 1.0 - target_acc
    return float(fr)


def _compute_losses(model, loader, device, target_class):
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    losses = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            mask = y == target_class
            if mask.sum().item() == 0:
                continue
            x = x[mask]
            y = y[mask]
            logits = model(x)
            l = loss_fn(logits, y).cpu().numpy()
            losses.extend(l)
    return np.asarray(losses)


def compute_fs(model, train_loader, test_loader, device, target_class):
    try:
        from sklearn import linear_model, model_selection
    except Exception as e:
        print(f"[Warn] sklearn unavailable for Fs calculation: {e}")
        return 0.0

    member_losses = _compute_losses(model, train_loader, device, target_class)
    nonmember_losses = _compute_losses(model, test_loader, device, target_class)

    if len(member_losses) < 5 or len(nonmember_losses) < 5:
        return 0.0

    n = min(len(member_losses), len(nonmember_losses))
    np.random.shuffle(member_losses)
    np.random.shuffle(nonmember_losses)

    x = np.concatenate([nonmember_losses[:n], member_losses[:n]]).reshape(-1, 1)
    y = np.array([0] * n + [1] * n)

    attack = linear_model.LogisticRegression(max_iter=1000)
    cv = model_selection.StratifiedShuffleSplit(n_splits=10, random_state=42)
    scores = model_selection.cross_val_score(attack, x, y, cv=cv, scoring="accuracy")
    mia_acc = scores.mean()
    fs = abs(0.5 - mia_acc)
    return float(fs)
