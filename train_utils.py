import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from temporal_averaging import temporal_avg


def train_model(data_loader, val_data_loader, epochs, optimizer, criterion,
                model, use_temporal_avg=False, device=torch.device("cuda:0"),
                temporal_window=3):
    training_loss, training_accuracy = [], []
    validation_loss, validation_accuracy = [], []
    model.train()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        csamp, closs = 0, 0
        for data, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            if use_temporal_avg:
                data, labels = temporal_avg(data, labels, temporal_window)
            data = data.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            predictions = model(data)
            _, pred = torch.max(predictions, dim=1)
            csamp += pred.eq(labels).sum().item()
            loss = criterion(predictions, labels)
            closs += loss.item()
            loss.backward()
            optimizer.step()

        tloss, tacc, _, _ = test(
            val_data_loader, criterion, model, device,
            use_temporal_avg, temporal_window
        )
        validation_accuracy.append(tacc)
        training_accuracy.append(csamp / len(data_loader.dataset))
        validation_loss.append(tloss)
        training_loss.append(closs / len(data_loader))

    history = {
        'training_loss': training_loss,
        'training_accuracy': training_accuracy,
        'validation_loss': validation_loss,
        'validation_accuracy': validation_accuracy,
    }
    return history


def test(data_loader, criterion, model, device, use_temporal_avg=False,
         temporal_window=3):
    predictions_cf, y_test = [], []
    model.eval()
    csamp, closs = 0, 0
    with torch.no_grad():
        for data, labels in data_loader:
            if use_temporal_avg:
                data, labels = temporal_avg(data, labels, temporal_window)
            y_test.extend(labels.cpu().numpy())
            data = data.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)
            predictions = model(data)
            _, pred = torch.max(predictions, dim=1)
            predictions_cf.extend(pred.cpu().numpy())
            csamp += pred.eq(labels).sum().item()
            loss = criterion(predictions, labels)
            closs += loss.item()
    return closs / len(data_loader), csamp / len(data_loader.dataset), predictions_cf, y_test
