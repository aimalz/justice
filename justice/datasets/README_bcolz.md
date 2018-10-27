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


## Downloading bcolz from teammates

To comply with the license agreement, make sure that all team members have Kaggle
accounts and have formally agreed to the [rules](https://www.kaggle.com/c/PLAsTiCC-2018/rules).
We believe that privately sharing this preprocessed data is in accordance with the
license (and is only being done to save us time).

<details><summary>Click to view a copy of the "Competition Data" section.</summary>
<p>

### 7. COMPETITION DATA.
"Competition Data" means the data or datasets available from the Competition Website for
the purpose of use in the Competition, including any prototype or executable code provided
on the Competition Website.

A. Data Access and Use. Unless otherwise restricted under the Competition Specific Rules
above, after your acceptance of these Rules, you may access and use the Competition Data
for the purposes of the Competition, participation on Kaggle Website forums, academic
research and education, and other non-commercial purposes.

B. Data Security. You agree to use reasonable and suitable measures to prevent persons who
have not formally agreed to these Rules from gaining access to the Competition Data. You
agree not to transmit, duplicate, publish, redistribute or otherwise provide or make
available the Data to any party not participating in the Competition. You agree to notify
Kaggle immediately upon learning of any possible unauthorized transmission or unauthorized
access of the Data and agree to work with Kaggle to rectify any unauthorized transmission.
You agree that participation in the Competition will not be construed as having or being
granted a license (expressly, by implication, estoppel, or otherwise) under, or any right
of ownership in, any of the Data.

C. External Data. Unless otherwise expressly stated on the Competition Website, you may
not use data other than the Competition Data to develop and test your models and
Submissions. Competition Sponsor reserves the right to disqualify any Participant who
Competition Sponsor discovers has undertaken or attempted to undertake the use of data
other than the Competition Data, or who uses the Competition Data other than as permitted
by the Competition Website and these Rules.

</p>
</details>

 1. Get the magnet link and key file from your teammates.
 1. Download the magnet link. Transmission (https://transmissionbt.com/) is a decent
    client for OS X and Linux if you need a recommendation. Press Ctrl+U and paste in the
    magnet link.
 1. If you don't already have the `openssl` binary, run `apt install -y openssl`.
 1. Run the following, replacing the key file and encrypted file as appropriate.

```
python -m justice.datasets.extract_downloaded \
    --key-file datadist_keyfile.key \
    --encrypted-file plasticc_bcolz.tar.enc
```
