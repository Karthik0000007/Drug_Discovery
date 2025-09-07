import numpy as np
import torch
import torch.nn as nn

# ---------------------------
# Training loop
# ---------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    cnt = 0
    for batch in loader:
        smiles = batch['smiles'].to(device)
        seq = batch['seq'].to(device)
        aff = batch['aff'].to(device).squeeze(1)
        pred = model(smiles, seq)
        loss = nn.functional.mse_loss(pred, aff)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * smiles.size(0)
        cnt += smiles.size(0)
    return total_loss / cnt

def eval_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in loader:
            smiles = batch['smiles'].to(device)
            seq = batch['seq'].to(device)
            aff = batch['aff'].to(device).squeeze(1)
            pred = model(smiles, seq)
            preds.append(pred.cpu().numpy())
            trues.append(aff.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return trues, preds