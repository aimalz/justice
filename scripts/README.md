First, download the Kaggle datset files. If you download files through the API tool or manually, place

* `test_set_metadata.csv`
* `test_set.csv`
* `training_set_metadata.csv`
* `training_set.csv`

in the `data/` folder. (If you don't want to move them, use `cp -s` to make symlinks
instead.)

If you downloaded the zip file (click "Download all"), run,

```sh
cd data/
unzip ~/Downloads/all.zip
chmod 644 *.csv
```

Then run ingestion scripts,

```sh
cd data/
sqlite3 plasticc_training_data.db < ../scripts/create_training_set.sql
sqlite3 plasticc_test_data.db < ../scripts/create_test_set.sql
```
