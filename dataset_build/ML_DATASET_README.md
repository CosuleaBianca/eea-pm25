# ML-Ready Dataset

The processed ML dataset (`ml_ready_dataset_full_realistic.csv`, ~1.6GB) is not included.

**Download from Hugging Face**: [LINK TO BE ADDED]

## Alternative: Regenerate from raw data
```bash
python dataset_build/src/process_data.py
python dataset_build/src/prepare_ml_dataset.py
python dataset_build/src/coverage_only_v6.py
python dataset_build/src/dataset_full_realistic_v6.py
```
