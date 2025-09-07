import os, math, argparse, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import DeepDTAModel
from data_loading import prepare_data
from tokenizers_and_datasets import build_vocab, DtaDataset
from train import train_epoch, eval_model
from utilities import set_seed, rmse, mae, pearsonr_np, spearmanr_np, concordance_index

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='CSV with columns smiles,sequence,affinity (optionally drug_id,target_id)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-sml-len', type=int, default=120)
    parser.add_argument('--max-prot-len', type=int, default=1000)
    parser.add_argument('--emb-dim', type=int, default=128)
    parser.add_argument('--conv-out', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--split', type=str, default='random', choices=['random','cold_drug','cold_target'])
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--ic50-nanomolar', action='store_true', help='If set, convert affinity (IC50 in nM) to pIC50 before training.')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--out', type=str, default='models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading data from {args.data} ...")
    df = pd.read_csv(args.data)
    required = ['smiles','sequence','affinity']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"CSV must contain column '{r}'")

    # optional conversion: IC50 (nM) -> pIC50
    if args.ic50_nanomolar:
        # pIC50 = -log10(IC50_M) ; IC50_M = IC50_nM * 1e-9 -> pIC50 = -log10(IC50_nM * 1e-9) = 9 - log10(IC50_nM)
        print("Converting IC50 (nM) to pIC50 ...")
        def to_pic50(v):
            try:
                v = float(v)
                if v <= 0:
                    return np.nan
                return 9.0 - math.log10(v)
            except:
                return np.nan
        df['affinity'] = df['affinity'].apply(to_pic50)
        df = df.dropna(subset=['affinity']).reset_index(drop=True)

    # clean: drop NA rows in sequences/smiles
    df = df.dropna(subset=['smiles','sequence','affinity']).reset_index(drop=True)
    print(f"Total examples: {len(df)}")

    # prepare splits
    train_df, val_df, test_df = prepare_data(df, split=args.split, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed)
    print(f"Train / Val / Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}")

    # build vocabularies
    sml_stoi, sml_itos = build_vocab(list(train_df['smiles']))
    prot_stoi, prot_itos = build_vocab(list(train_df['sequence']))
    print(f"Drug vocab size: {len(sml_itos)} (including PAD/UNK). Protein vocab size: {len(prot_itos)}")

    train_ds = DtaDataset(train_df, sml_stoi, prot_stoi, max_sml_len=args.max_sml_len, max_prot_len=args.max_prot_len)
    val_ds = DtaDataset(val_df, sml_stoi, prot_stoi, max_sml_len=args.max_sml_len, max_prot_len=args.max_prot_len)
    test_ds = DtaDataset(test_df, sml_stoi, prot_stoi, max_sml_len=args.max_sml_len, max_prot_len=args.max_prot_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device(args.device)
    model = DeepDTAModel(vocab_drug=len(sml_itos), vocab_prot=len(prot_itos),
                         emb_dim=args.emb_dim, conv_out=args.conv_out, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_rmse = 1e9
    patience = args.patience
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        st = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        trues_val, preds_val = eval_model(model, val_loader, device)
        val_rmse = rmse(trues_val, preds_val)
        scheduler.step(val_rmse)

        print(f"[Epoch {epoch}] TrainLoss: {train_loss:.4f} ValRMSE: {val_rmse:.4f} Time: {time.time()-st:.1f}s")

        # save best
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            out_path = os.path.join(args.out, f'best_epoch{epoch}_rmse{val_rmse:.4f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'sml_itos': sml_itos,
                'prot_itos': prot_itos,
                'args': vars(args),
            }, out_path)
            print(f"Saved best model to {out_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break

    # Load best checkpoint (last saved) for final eval
    chkpts = [os.path.join(args.out, f) for f in os.listdir(args.out) if f.endswith('.pt')]
    if len(chkpts) == 0:
        print("No checkpoint found; evaluating current model.")
    else:
        chk = sorted(chkpts, key=os.path.getmtime)[-1]
        print(f"Loading checkpoint {chk} for final evaluation.")
        dd = torch.load(chk, map_location=device)
        model.load_state_dict(dd['model_state'])

    # final evaluations
    trues_train, preds_train = eval_model(model, train_loader, device)
    trues_val, preds_val = eval_model(model, val_loader, device)
    trues_test, preds_test = eval_model(model, test_loader, device)

    def summarize(name, y_true, y_pred):
        print(f"=== {name} ===")
        print(f"RMSE: {rmse(y_true, y_pred):.4f}")
        print(f"MAE:  {mae(y_true, y_pred):.4f}")
        print(f"Pearson r: {pearsonr_np(y_true, y_pred):.4f}")
        print(f"Spearman r: {spearmanr_np(y_true, y_pred):.4f}")
        print(f"Concordance Index: {concordance_index(y_true, y_pred):.4f}")
        print("")

    summarize("Train", trues_train, preds_train)
    summarize("Val", trues_val, preds_val)
    summarize("Test", trues_test, preds_test)

    # save predictions on test set
    out_preds = pd.DataFrame({
        'true': trues_test,
        'pred': preds_test
    })
    out_preds.to_csv(os.path.join(args.out, 'test_predictions.csv'), index=False)
    print(f"Saved test predictions to {os.path.join(args.out,'test_predictions.csv')}")

if __name__ == '__main__':
    main()