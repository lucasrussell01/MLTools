# Production

**IN DEVELOPMENT**

`PreProcess.py` will select same sign or opposite sign events from the `parquet` files that are outputed by HiggsDNA.

`Stitching.py` will stitch together the inclusive and exclusive DY samples and modify the central weight.

`NNReweight.py` will reweight ggH and VBF contributions to expected cross sections, and all 3 categories (Higgs, Taus, Fakes) so that they have equal importance for training.

`ShuffleMerge.py` will add truth lables and mix different classes for training.

