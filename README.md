# Machine Learning Framework 

This repository provides a framework for various Machine Learning based tasks. 

The general structure is:
- **Production**: Production, Selection and Reweighting of tuples.
- **Analysis**: Analyse properties of tuples. Shuffle and Merge for training (including truth labelling).
- **Training**: Dataloading and Training. Implementation of various models.
- **Evaluation**: Evaluate model performance.

This `main` branch will contain a generalised framework, and tools/models.

Different branches are being developed for specific tasks, currently:
- `SignalClassifier`: Multiclass classification for CP Analysis (Higgs, Genuine Tau, and Background)
- `FeatureImportance`: Feature importance studies (eg DeepTau)
