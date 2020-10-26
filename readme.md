# README

## Background

This git repository contains the code and data files for the artefact of my master's thesis titled "ARCHITECTURE-INDEPENDENT MATCHING OF STRIPPED BINARY CODE FILES USING BERT AND A SIAMESE NEURAL NETWORK" from the University of Jyväskylä.

## Prerequisites

The scripts included in this repository require the installation of the following:

- BERT (https://github.com/google-research/bert)
- Keras (https://keras.io)
- Tensorflow (https://www.tensorflow.org)

These scripts were tested to run on a Windows 10 installation with tensorflow-gpu version 2.0.0 and Keras version 2.3.1. 


## Thesis artefact

The thesis artefact consists of four different stages  as noted in the image below.

![Thesis artefact stages](/img/artefact_stages.png)

In particular, the dataset preparation, ISA model learning and code instruction embeddings stages require an understanding of how to use the BERT code. It is highly recommended that you familiarize yourself with the README file at https://github.com/google-research/bert/blob/master/README.md before proceeding with the scripts in this repository.

## Overview of thesis datasets

Below is a table of the various datasets used in the thesis artefact. These are referenced in the steps below.

![Thesis artefact datasets](/img/DataSetsTable.png)


## Thesis artefact validation

If you want to quickly validate the similarity detection portion of the artefact, please focus on the ‘Similarity detection: Test’ and ‘Fuzzy hashing comparison’ sections below. These sections contain commands and associated screen shots of the output.



## Thesis stages

### Dataset preparation

In the folder 1\_dataset\_preparation, you will find the following:

- TODO: a link to data sets 1 & 2
- bin\_vocab\_builder.py
- bin2txt.bulk.py
- gradu\_data\_prep.py

To prepare binary data to pre-train BERT, do the following:

1) Create a vocabulary file using bin\_vocab\_builder.py (requires edit to output location for armhf\_vocab.txt)
2) Create text files from binary code sections by running bin2txt.bulk.py on datasets 1 and 2 (requires edit to locations for input code section files and output files)
3) Create one big text file for each type using gradu\_data\_prep.py (requires edit to locations for input text files and output files)
4) Create shards for each type using the following command edited for your environment:

python C:\Users\Kenneth\Documents\GitHub\bert\shard\_file.py ^
  --input-file "C:\gradu2\armhf.txt" ^
  --shards-directory "C:\gradu2\shards" ^
  --shard-size

For more information on shard\_file.py, please consult the BERT documentation.

5) Create tfrecord files each type using the following command edited for your environment:

python C:\Users\Kenneth\Documents\GitHub\bert\create\_pretraining\_data.py ^
  --input\_file=C:\gradu2\shards\shard\_\*.txt ^
  --output\_file=C:\gradu2\tf\_examples128.tfrecord\* ^
  --vocab\_file=C:\gradu2\armhf\_vocab.txt ^
  --do\_lower\_case=True ^
  --max\_seq\_length=128 ^
  --max\_predictions\_per\_seq=20 ^
  --masked\_lm\_prob=0.15 ^
  --random\_seed=12345 ^
  --dupe\_factor=5

For more information on create\_pretraining\_data.py, please consult the BERT documentation.

### ISA model learning

In the folder 2\_ISA\_model\_learning, you will find:

- TODO: a link to BERT\_train\_data

To pre-train BERT, use the following command customized for the downloaded BERT\_train\_data in your environment:

python C:\Users\Kenneth\Documents\GitHub\bert\run\_pretraining.py ^
  --input\_file=D:\Gradu\rasp\_tf\_examples.tfrecord ^
  --output\_dir=D:\Gradu\pretraining\_output\_rasp ^
  --do\_train=True ^
  --do\_eval=True ^
  --bert\_config\_file=C:\Users\Kenneth\Documents\GitHub\bert\bert\_config.json ^
  --init\_checkpoint=D:\Gradu\pretraining\_output\_5\model.ckpt-100000 ^
  --train\_batch\_size=32 ^
  --max\_seq\_length=128 ^
  --max\_predictions\_per\_seq=20 ^
  --num\_train\_steps=100000 ^
  --num\_warmup\_steps=10 ^
  --learning\_rate=2e-5

Pre-training should be run using data sets 1 & 2 over multiple rounds until the desired accuracy is attained. For more information on run\_pretraining.py, please consult the BERT documentation.


### Code instruction embeddings

In the folder 3\_code\_instruction\_embeddings, you will find:

- batch\_embeddings\_from\_BERT.py
- TODO: link to pre\_trained\_BERT.zip

To create the embedding files needed to train the Siamese network, do the following:

1) unzip the file pre\_trained\_BERT.zip
2) in batch\_embeddings\_from\_BERT.py, change paths for your environment on lines 7 and 12 as noted in script comments
3) run batch\_embeddings\_from\_BERT.py multiple times to create the following data sets: 5, 6, 7, 8, 9, 10

For questions on editing batch\_embeddings\_from\_BERT.py, please consult the BERT documentation for extract\_features.py.


### Code file vector space representation and similarity detection

In the folder 4a\_similarity\_detection, you will find two folders: train and test. Test is set up to validate the already trained Siamese model from the thesis. Train is set to train a new model of the Siamese network.

#### Similarity detection: Train
In the train folder you will find the following:
- TODO: link to embedding files needed to training the Siamese network (Data sets 5, 6, 7)
- one folder containing the scripts needed to train a new model of the Siamese network.

Follow the instructions below to train a new model:

1) unzip the train\_data.zip file
2) open the train\_scripts folder
3) edit the GRADU\_Siamese\_Train\_GEN.py file to include the paths to the corresponding directories in the unzipped data (lines 11, 12, 15, 18)
4) edit line 13 of siamese\_train.py to include the path to the directory containing the unzipped training data
5) run siamese\_train.py to train the model. The trained model is saved as Siamese\_RNN\_01.h5. This can be changed in line 82.

#### Similarity detection: Test
In the test folder you will find the following:
- TODO: link to embedding files needed to evaluate the Siamese network (Data sets 8, 9, 10)
- one folder containing:
	- the scripts needed to evaluate the accuracy of the already trained Siamese model from the thesis
	- the trained model named Siamese\_RNN\_13.h5

Follow the instructions below to evaluate the accuracy of the Siamese model from the thesis:

1) unzip the test\_data.zip file
2) open the test\_scripts folder
3) edit the GRADU\_Siamese\_Test\_GEN.py file to include the paths to the corresponding directories in the unzipped data (lines 11, 12, 15, 18)
4) edit line 13 of siamese\_evaluate.py to include the path to the directory containing the unzipped testing data
5) edit line 13 of siamese\_predict.py to include the path to the directory containing the unzipped testing data
6) run siamese\_evaluate.py to evaluate the model using the Keras evaluate function
7) run siamese\_predict.py to evaluate the model using the Keras predict function

Running siamese\_predict.py will produce the following output:

![Predict output screenshot](/img/Predict_Output.png)


#### Fuzzy hashing comparison

In the folder 4b\_fuzzy\_hashing\_comparison, you will find a zip file named fuzzy\_hash\_dataset.zip. It contains data sets 11, 12 & 13.

The thesis comparison was run using version 3.1\_2 of sdhash and version 2.14.1 of ssdeep. These will need to be installed.

The following commands were run from the folder containing the dataset to conduct the comparison:

	ssdeep -s *.code > database.ssd
	ssdeep -m database.ssd *.code -s

![SSDEEP output screenshot](/img/SSDEEP_screenshot.png)


	sdhash *.code > code.sdbf
	sdhash --compare code.sdbf

![SDHASH output screenshot](/img/SDHASH_screenshot.png)
