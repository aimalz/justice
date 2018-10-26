# Instructions for bcolz format

## Generating data from Kaggle CSV files

Using zsh,

```sh
for i in data/(training|test)_set_metadata.csv data/(training|test)_set.csv; do
    python3 -m justice.datasets.plasticc_bcolz --source-file $i;
done
```

(The above invocation splits metadata and test set csv generation,
so if there are errors it will fail faster.)


