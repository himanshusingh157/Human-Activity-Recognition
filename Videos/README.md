# Download the dataset
The dataset is in 3 parts: [part 1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001), [part 2](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002), [part 3](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003)\
Download the 3 files, move into "videos" folder
Combine the zip file into one big zip file
```
cat ucf101_jpegs_256.zip.00* > ./abc.zip
```
Unzip this big file
```
unzip ./abc.zip
```

# Traning model
To train the particular model, run
## 3D-CNN
```
python 3D_CNN.py
```  
## ResNet-LSTM
```
python ResNet_LSTM.py
```  


