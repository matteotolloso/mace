## Evoluzione di AU, EU durante il training

### AU, EU ed errore durante il training con standard mean-variance ensemble (logvar predictor)

![](outputs/plots/unc_vs_epoch_train.png)


### Analisi per ensemble member: gli spikes sono dati principalmente dai singoli membri e dalle singole configurazioni.

![](outputs/plots/au_members_configs_train.png)

### Con sofplus i risultati migliorano ma forse serve aggiungere clipping o altro.

![](outputs/plots/au_members_configs_train-mv-sp.png)

### Di conseguenza anche la media delle AU migliora

![](outputs/plots/unc_vs_epoch_train-mv-sp.png)


### Il reliability diagram del best model (selezionato come miglior errore su validation con ensemble) va bene (ma non benissimo)

![](outputs/plots/unc_vs_error.png)

### Singoli sistemi

![](tmp/TODO.png)

### Prime due righe: minore incertezza totale, seconde due righe: media incertezza totale, terza riga: maggiore incertezza totale.

![](outputs/plots/member_pred_var_grid.png)




### task

- System‑OOD (nuovi sistemi)

- Energy‑OOD (stessi sistemi, energia diversa)


### 2 regimi di training

- Train/test semplice (single‑fidelity)

- Pretrain -> Fine‑tune (multi‑fidelity): Pretrain su livello low‑fidelity (DFT) grande. Fine‑tune su high‑fidelity (CC) piccolo (e calibration eventualmente).