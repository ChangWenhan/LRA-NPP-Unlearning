import torch


def evaluate_at_ag(model, dataloader, device, target_class):
    model.eval()
    target_total = 0
    target_correct = 0
    other_total = 0
    other_correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            mask_target = labels == target_class
            mask_other = labels != target_class

            target_total += mask_target.sum().item()
            target_correct += ((preds == labels) & mask_target).sum().item()

            other_total += mask_other.sum().item()
            other_correct += ((preds == labels) & mask_other).sum().item()

    at = target_correct / target_total if target_total > 0 else 0.0
    ag = other_correct / other_total if other_total > 0 else 0.0
    return at, ag
