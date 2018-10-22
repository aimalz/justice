# Instructions for bcolz format

## Generating data from Kaggle CSV files

Using zsh,

```sh
for i in data/(training|test)_set(|_metadata).csv; do
    python3 -m justice.datasets.plasticc_bcolz --source-file $i;
done
```


