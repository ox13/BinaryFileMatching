import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import jsonlines
import json
import random
#from keras.preprocessing.sequence import pad_sequences as pad 

# PATH TO RASP DOCUMENT VECTOR FILES
rasp_path = "C:\\..\\..\\20_rasp_EMBED_match_ess\\"
rasp_no_match_path = "C:\\..\\..\\20_rasp_EMBED_no_match_ess\\"

# PATH TO ARMHF DOCUMENT VECTOR FILES
armhf_path = 'C:\\..\\..\\20_armhf_EMBED_match_ess\\'

# path to JSON files
json_path = "C:\\..\\..\\bin_jsons\\"

match_counter = []
for i in range(1, 21):
  match_counter.append(i)


class Config():

    def __init__(self):

        pass

    # num_epochs =10
    batch_size =1



class BinaryFileDocvec(object):
    
    """
    Implements a data loader that reads tha data and creates a data generator 
    which can be directly used to train the model
    """
    
    def __init__(self,root_dir=None):     
        self.root_dir = root_dir
        
    def load_samples(self,csv_file):
        
        """
        function to read a csv file and create a list of samples of format
        
        """
        # Read the csv file
        data = pd.read_csv(os.path.join(self.root_dir,'data_files',csv_file))
        
        # Get the filename contained in the first column
        file_names = list(data.iloc[:,0])
        
        samples=[]
        for samp in zip(file_names):
            samples.append([samp])
        return samples
        
        
    def shuffle_data(self,data):
        data = shuffle(data)#,random_state=2)
        return data
    

    def matching_file(self,filename):
        # one way match:  armhf in rasp out (despite variable names below)
        file_orig = os.path.basename(filename)
        file_orig = os.path.splitext(file_orig)[0]
        file_orig = os.path.splitext(file_orig)[0]
        file_orig = os.path.splitext(file_orig)[0]
        file_orig = os.path.splitext(file_orig)[0]
                
                # # open the rasp json file as read-only
        with open(json_path + "raspbian.json", "r") as read_file:
            rasp_data = json.load(read_file)
        # open the armhf JSON file as read-only
        with open(json_path + "armhf.json", "r") as hf_read_file:
            armhf_data = json.load(hf_read_file)

        for i in armhf_data:
            if file_orig == i["filehash"]:
                rasp = i["filename"]
                rasp = rasp.split("/")[-1]
                rasp_db = i["deb_package"]
                rasp_db = rasp_db.split("_")[0:-1]
                for v in rasp_data:
                    armhf = v["filename"]
                    armhf = armhf.split("/")[-1]
                    armhf_db = v["deb_package"]
                    armhf_db = armhf_db.split("_")[0:-1]
                    armhf_file_hash = v["filehash"]
                    if rasp == armhf and rasp_db == armhf_db:
                        armhf_match = armhf_file_hash   
                        file_match = rasp_path + armhf_match + ".code.txt.jsonl.docvec" 
                        return file_match
        
    def match_coord(self, filename):
        global match_counter
        #global counter_plus
        #print(match_counter)
        z =  match_counter[random.randint(0, 19)]
        #counter_plus += 1
        #print(z)
        if z % 2 == 0:
            match_file = self.matching_file(filename)
            label = 1
        else:
            num1 = random.randint(0, 39)
            for dirpath, subdirs, files in os.walk(rasp_no_match_path):
                match_file = files[num1]
                match_file = rasp_no_match_path + match_file
                label = 0
        return match_file, label


    def data_generator(self,data,batch_size=10,shuffle=True):              
        """
        Yields the next training batch.
        """
        num_samples = len(data)
        if shuffle:
            data = self.shuffle_data(data)
        while True:   
            for offset in range(0, num_samples, batch_size):
#                print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                batch_samples = data[offset:offset+batch_size]
                # Initialise X_train and y_train arrays for this batch
                X_train_pre = []
                Xb_train_pre = []

        
                # For each example
                for batch_sample in batch_samples:
#                    print (batch_sample)
                    # Load image (X)
#                    x = batch_sample[0]
                    img_name = batch_sample[0]
                    img_name = img_name[0]
                    armhf_name, auto_label = self.match_coord(img_name)
                                     
                    img = jsonlines.open(img_name)
                    img2 = jsonlines.open(armhf_name)
#                   print (img.shape)
                                    
                    X_train_pre.append(img)
                    reader = X_train_pre[-1]
                    X_base = reader.read(type=list)

                    Xb_train_pre.append(img2)
                    reader = Xb_train_pre[-1]
                    X_match = reader.read(type=list)

        
                # Make sure they're numpy arrays (as opposed to lists)
                X_base = np.array(X_base)
                base_length = len(X_base)
                X_base = X_base.reshape(base_length, 1, 768)

                X_match = np.array(X_match)
                match_length = len(X_match)
                X_match = X_match.reshape(match_length, 1, 768)

                # padding the files to be of equal number of time series
                new_sents = [] 
                new_sent =[0] * 768
                label = []
                #label.append(auto_label)
                # adding # of classes here
                #label.append(2)
                #label = np.array(label)
                
                if base_length > match_length:
                    for k in range(base_length - match_length):
                        new_sents.append(new_sent)
                    for l in range(base_length):    
                        label.append([auto_label])
                        #label.append(label)
                    new_sents = np.array(new_sents)
                    new_sents = new_sents.reshape((base_length - match_length), 1, 768)                    
                    X_match = np.concatenate((X_match, new_sents), axis=0)
                    label = np.array(label)
                    #change here also: label_array from label
                    #label = np.concatenate((label_array, label_pad), axis=0)
                elif match_length > base_length:                    
                    for k in range(match_length - base_length):
                        new_sents.append(new_sent)
                    for l in range(match_length):    
                        label.append([auto_label])
                        #label.append(label)
                    new_sents = np.array(new_sents)
                    new_sents = new_sents.reshape((match_length - base_length), 1, 768)
                    X_base = np.concatenate((X_base, new_sents), axis=0)
                    label = np.array(label)
                else:
                    nothing = 0
       
                # The generator-y part: yield the next training batch            
                yield [X_base, X_match], label
    




            
