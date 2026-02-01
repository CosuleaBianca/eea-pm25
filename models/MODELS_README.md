# Trained Models

Pre-trained models are not included due to size. Download from:

**Hugging Face**: [LINK TO BE ADDED]

## To retrain models:
```bash
# Ensure data files are in place, then:
python train_models/global/lr_global.py
python train_models/global/rf_global.py
python train_models/global/xgb_global.py
python train_models/global/lgb_global.py
python train_models/global/gam_global.py
python train_models/global/lstm_global.py
python train_models/global/lstm_global_residual.py
python train_models/global/lstm_global_attention.py
python train_models/global/lstm_cnn.py
```
