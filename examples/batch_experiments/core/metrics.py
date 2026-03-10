import torch
from tqdm.auto import tqdm


def evaluate_at_ag(model, dataloader, device, target_class, show_progress=False, progress_desc=None):
    model.eval()
    target_total = 0
    target_correct = 0
    other_total = 0
    other_correct = 0

    iterator = dataloader
    if show_progress:
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc=progress_desc or "Evaluate At/Ag",
            leave=False,
        )

    with torch.no_grad():
        for images, labels in iterator:
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

            if show_progress:
                at_now = (target_correct / target_total) if target_total > 0 else 0.0
                ag_now = (other_correct / other_total) if other_total > 0 else 0.0
                iterator.set_postfix(At=f"{at_now:.4f}", Ag=f"{ag_now:.4f}")

    at = target_correct / target_total if target_total > 0 else 0.0
    ag = other_correct / other_total if other_total > 0 else 0.0
    return at, ag
