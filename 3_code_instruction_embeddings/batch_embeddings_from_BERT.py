import os

# Used to initiate creation of embeddings from several text files in a given directory using BERT model denoted in line 12
# NOTE: Change paths for file locations in 3 locations: Lines 7 (1 path for input files) and 12 (2 paths - 1 input, 1 output)

count = 0
path="E:\\Gradu\\DataSet\\Match\\MatchedFiles_txt\\Expanded_rasp_no_match\\"

for dirpath, subdirs, files in os.walk(path):
    for filename in files:
        if filename != '.DS_Store.txt':
            os.system("python C:\\Users\\Kenneth\\Documents\\GitHub\\bert\\extract_features.py --input_file=E:\\Gradu\\DataSet\\Match\\MatchedFiles_txt\\Expanded_rasp_no_match\\" + filename + " --output_file=E:\\Gradu\\DataSet\\Match\\MatchedFiles_json_essential\\Train\\exp_rasp_EMBED_no_match_ess\\" + filename + ".jsonl.docvec --vocab_file=C:\\gradu2\\armhf_vocab.txt --bert_config_file=C:\\bert_config.json --init_checkpoint=D:\\Gradu\\pretraining_output_rasp_2\\model.ckpt-100000 --layers=-2 --max_seq_length=128 --batch_size=8")
            print("%s File Embedding Successfull" % filename)
            count = count + 1
            print("%s file embeddings completed. " % count)
