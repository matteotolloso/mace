controlla che le unità di misura delle energie siano coerenti tra ani e mace

Dataset: 3 esperimenti

2 tipologie di esperimenti:
- train/test semplice con uncertainty calibrata

    possibili scenari:
    - sistemi nuovi rispetto a training.
    - stessi sistemi ma energia diversa.

- pre-train fine tuning (Multi‑fidelity + UQ calibrata)

    possibili scenari:
    - sistemi nuovi rispetto a training.
    - stessi sistemi ma energia diversa.


protocollo di training: 
- massimo numero di epoche per tutti i modelli
- selezione del miglior modello in base al validation set
- costruzione dell'ensemble con i migliori modelli selezionati

plotting e analisi delle incertezze:
- è interessnte l'andamento delle incertezze , ma se presa come misura assoluta è possibile che una delle due domini. Quindi provare a plottare entrambe dopo una regressione isotonica indipendente. (e anche la total uncertainty)

Plot epoch-uncertainty: forse non ha molto senso perchè allineare i membri alla stessa epoca è una forzatura senza giustificatione teorica.