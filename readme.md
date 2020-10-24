# README

## Background

This git repository contains the code and data files for my master's thesis titled "ARCHITECTURE-INDEPENDENT MATCHING OF STRIPPED BINARY CODE FILES USING BERT AND A SIAMESE NEURAL NETWORK" from the University of Jyväskylä.

## Prerequisites

The scripts included in this repository require the installation of the following:

- BERT (https://github.com/google-research/bert)
- Keras (https://keras.io)
- Tensorflow (https://www.tensorflow.org)

In particular, Dataset preparation, ISA model learning and Code instruction embeddings require an understanding of how to use the BERT code. It is highly reccommended that you familiarize yourself with the README file at https://github.com/google-research/bert/blob/master/README.md.

## Overview of thesis datasets

Below is a table of the various datasets used in the thesis artefact. These are referenced in the steps below.

Data Set	Derived From	                    Format	            Type	Count	Match 	Used For
1	        Debian repositories	                code sections	    armhf	~3700	NA	    Pre-training
2	        Raspbian repositories	            code sections	    rasp	~300	NA	    Fine tuning
3	        Data set 1	                        code sections	    armhf	300	    Yes	    creating other data sets
4	        Data set 2	                        code sections	    rasp	300	    Yes	    creating other data sets
5	        Data set 3	                        embedding files	    armhf	253	    Yes	    Siamese network training
6	        Data Set 4	                        embedding files	    rasp	253	    Yes	    Siamese network training
7	        Raspbian repositories	            embedding files	    rasp	297	    No	    Siamese network training
8	        Data set 3	                        embedding files	    armhf	20	    Yes	    Siamese network validation
9	        Data set 4	                        embedding files	    rasp	20	    Yes	    Siamese network validation
10	        Raspbian repositories	            embedding files	    rasp	40	    No	    Siamese network validation
11	        Data sets 3 & 8	                    code sections	    armhf	20	    Yes	    SSDEEP & SDHASH testing
12	        Data sets 4 & 9	                    code sections	    rasp	20	    Yes	    SSDEEP & SDHASH testing
13	        Raspbian repositories & Data set 10	code sections	    rasp	40      No	    SSDEEP & SDHASH testing


## Dataset preparation

In the folder 1_dataset_preparation, you will find the following:

- TODO: a link to data sets 1 & 2
- bin_vocab_builder.py
- bin2txt.bulk.py
- gradu_data_prep.py

To prepare binary data to pre-train BERT, do the following:

1) Create a vocabulary file using bin_vocab_builder.py (requires edit to output location for armhf_vocab.txt)
2) Create text files from binary code sections by running bin2txt.bulk.py on datasets 1 and 2 (requires edit to locations for input code section files and output files)
3) Create one big text file for each type using gradu_data_prep.py (requires edit to locations for input text files and output files)
4) Create shards for each type using the following command edited for your environment:

python C:\Users\Kenneth\Documents\GitHub\bert\shard_file.py ^
  --input-file "C:\gradu2\armhf.txt" ^
  --shards-directory "C:\gradu2\shards" ^
  --shard-size

For more information on shard_file.py, please consult the BERT documentation.

5) Create tfrecord files each type using the following command edited for your environment:

python C:\Users\Kenneth\Documents\GitHub\bert\create_pretraining_data.py ^
  --input_file=C:\gradu2\shards\shard_*.txt ^
  --output_file=C:\gradu2\tf_examples128.tfrecord* ^
  --vocab_file=C:\gradu2\armhf_vocab.txt ^
  --do_lower_case=True ^
  --max_seq_length=128 ^
  --max_predictions_per_seq=20 ^
  --masked_lm_prob=0.15 ^
  --random_seed=12345 ^
  --dupe_factor=5

For more information on create_pretraining_data.py, please consult the BERT documentation.

## ISA model learning

In the folder 2_ISA_model_learning, you will find:

- TODO: a link to BERT_train_data

To pre-train BERT, use the following command customized for the downloaded BERT_train_data in your environment:

python C:\Users\Kenneth\Documents\GitHub\bert\run_pretraining.py ^
  --input_file=D:\Gradu\rasp_tf_examples.tfrecord ^
  --output_dir=D:\Gradu\pretraining_output_rasp ^
  --do_train=True ^
  --do_eval=True ^
  --bert_config_file=C:\Users\Kenneth\Documents\GitHub\bert\bert_config.json ^
  --init_checkpoint=D:\Gradu\pretraining_output_5\model.ckpt-100000 ^
  --train_batch_size=32 ^
  --max_seq_length=128 ^
  --max_predictions_per_seq=20 ^
  --num_train_steps=100000 ^
  --num_warmup_steps=10 ^
  --learning_rate=2e-5

Pre-training should be run using data sets 1 & 2 over multiple rounds until the desired accuracy is attained. For more information on run_pretraining.py, please consult the BERT documentation.


## Code instruction embeddings

In the folder 3_code_instruction_embeddings, you will find:

- batch_embeddings_from_BERT.py
- pre_trained_BERT.zip

To create the embedding files needed to train the Siamese network, do the following:

1) unzip the file pre_trained_BERT.zip
2) in batch_embeddings_from_BERT.py, change paths for your environment on lines 7 and 12 as noted in script comments
3) run batch_embeddings_from_BERT.py multiple times to create the following data sets: 5, 6, 7, 8, 9, 10

For questions on editing batch_embeddings_from_BERT.py, please consult the BERT documentation for extract_features.py.


## Similarity detection

In the folder 4a_similarity_detection, you will find two folders: train and test. Test is set up to validate the already trained Siamese model from the thesis. Train is set to train a new model of the Siamese network.

### Train
In this folder you will find the following:
- TODO: link to embedding files needed to training the Siamese network (Data sets 5, 6, 7)
- one folder containing the scripts needed to train a new model of the Siamese network.

Follow the instructions below to train a new model:

1) unzip the train_data.zip file
2) open the train_scripts folder
3) edit the GRADU_Siamese_Train_GEN.py file to include the paths to the corresponding directories in the unzipped data (lines 11, 12, 15, 18)
4) edit line 13 of siamese_train.py to include the path to the directory containing the unzipped training data
5) run siamese_train.py to train the model. The trained model is saved as Siamese_RNN_01.h5. This can be changed in line 82.

### Test
In this folder you will find the following:
- TODO: link to embedding files needed to evaluate the Siamese network (Data sets 8, 9, 10)
- one folder containing:
    - the scripts needed to evaluate the accuracy of the already trained Siamese model from the thesis
    - the trained model named Siamese_RNN_13.h5

Follow the instructions below to evaluate the accuracy of the Siamese model from the thesis:

1) unzip the test_data.zip file
2) open the test_scripts folder
3) edit the GRADU_Siamese_Test_GEN.py file to include the paths to the corresponding directories in the unzipped data (lines 11, 12, 15, 18)
4) edit line 13 of siamese_evaluate.py to include the path to the directory containing the unzipped testing data
5) edit line 13 of siamese_predict.py to include the path to the directory containing the unzipped testing data
6) run siamese_evaluate.py to evaluate the model using the Keras evaluate function
7) run siamese_predict.py to evaluate the model using the Keras predict function


## Fuzzy hashing comparison

In the folder 4b_fuzzy_hashing_comparison, you will find a zip file named fuzzy_hash_dataset.zip. It contains data sets 11, 12 & 13.

The thesis comparison was run using version 3.1_2 of sdhash and version 2.14.1 of ssdeep. These will need to be installed.

The following commands were run from the folder containing the dataset to conduct the comparison:

ssdeep -s *.code > database.ssd
ssdeep -m database.ssd *.code -s

sdhash *.code > code.sdbf
sdhash --compare code.sdbf