### how to download data from kaggle

!pip install --upgrade kaggle

## upload kaggle credentials 'kaggle.json' to wkdir

## open a new terminal

## cd ~/.kaggle

## mv ~/wkdir/kaggle.json ./

## chmod 600 ./kaggle.json

!kaggle competitions download -c widsdatathon2021

## unzip downloaded file
import zipfile
with zipfile.ZipFile( wkdir + 'widsdatathon2021.zip', 'r') as zip_ref:
    zip_ref.extractall(wkdir + '/data')
