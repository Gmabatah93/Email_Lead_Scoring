# PREPROCESS
- Opted to not do distributed preprocessing because of a small dataset

- for XGBoost
    - Encoded country_code column
    - checked for missingness (Filled NA with -999 if any: WHY?)
    - split Target Column


# TRAINING

## Splitting Strategy
- Train test, stratify y to ensure train and test sets have the same proportion 