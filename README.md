# Machine Learning Framework 

This repository provides a framework for various Machine Learning based tasks. 

The general structure is:
- **Production**: Production, Selection and Reweighting of tuples (including truth labelling)
- **Analysis**: Tools to analyse properties of produced tuples.
- **Training**: Dataloading and Training. Implementation of various models.
- **Evaluation**: Evaluate model performance.

This `main` branch will contain a generalised framework, and tools/models.

Different branches can be setup for specific tasks, currently:
- `SignalClassifier`: Multiclass classification for CP Analysis (Higgs, Genuine Tau, and Background)
