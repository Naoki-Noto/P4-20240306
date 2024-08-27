# P4-20240306
Python (3.10.13) was used as a language for this research, and used packages were keras (2.15.0), matplotlib (3.8.2), mordred (1.2.0), numpy (1.24.4), pandas (2.2.2), rdkit (2023.9.5), shap (0.44.1) seaborn (0.13.2) scikeras (0.12.0), scikit-learn (1.4.0), tensorflow (2.15.0).


# Table of Contents

Performance_comparison/

  • Make_descriptors: Code to generate descriptor sets.

  • data: Datasets used for ML, which were derived from Morgan fingerprint, MACCSKeys, RDKit descriptor and Mordred.

  • result: Results of ML.

  • HGB.py/NN.py/RF.py/Ridge.py/SVM.py: Code for constructing each ML model.

==============================================================================
  
TCI_reagents/

  • Make_descriptors: Code to generate descriptor sets.

  • data: Datasets used for ML.

  • result: Results of ML.

  • Analysis.py: Code to analyze the result of virtual screening.
  
  • HGB.py: Code for the virtual screening based on the HGB/Mordred model.
  
  • SHAP_scatter.ipynb: Code to generate SHAP scatter plots.
  
  • SHAP_waterfall.ipynb: Code to generate SHAP waterfall plots.
