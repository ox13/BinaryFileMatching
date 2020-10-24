import os
import io
from bitstring import ConstBitStream

count = 0
path = "/Users/kenlampinen/gradu_final/DataSet/CodeSections/raspbian/"
path_final = "/Users/kenlampinen/gradu_final/DataSet/BinaryAsText/raspTXTprocessed/"

for dirpath, subdirs, files in os.walk(path):
    for filename in files:
        x = filename
        s = ConstBitStream(filename=path + x)
        text_file = open(path_final + x + ".txt", "w")

        # add line return every 32 char
        def insert_newlines(string, every=32):
            lines = []
            for i in range(0, len(string), every):
                lines.append(string[i:i+every])
            return '\n'.join(lines)

        line_output = insert_newlines(s.bin)
        output = io.StringIO(line_output)
        saved_output = ""

        # add spacing for ARM "words" (12,4,4,1,4,1,2,4)
        for line in output.readlines():
            line = line[0:12] + " " + line[12:16] + " " + line[16:20] + " " + line[20:21] + " " + line[21:25] + " " + line[25:26] + " " + line[26:28] + " " + line[28:32] + "\n"
            saved_output = saved_output + str(line)

        # write to text file
        text_file.write(saved_output)
        text_file.close()



