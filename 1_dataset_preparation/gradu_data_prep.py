# import os
# path="/Users/kenlampinen/gradu_final/test/files/"
# count = 0

# for dirpath, subdirs, files in os.walk(path):
#     for filename in files:
#         with open('/Users/kenlampinen/gradu_final/test/files/compiled.txt', 'wb') as outfile:
#             #for fname in filename:
#             with open(filename) as infile:
#                 for line in infile:
#                     outfile.write(line)
#                 outfile.write("\n")

# This file combines all the text files in the directory into one large text file

import glob

read_files = glob.glob("/Users/kenlampinen/gradu_final/MATCHcode/raspTXTprocessed/*.txt")

with open("/Users/kenlampinen/gradu_final/MATCHcode/raspMATCHcombined.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())
            outfile.write("\n".encode())