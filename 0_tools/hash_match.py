import json
import os
import sys

# Tool to find the matching file name.
# Enter in the hash number of the file for which you want to find a match.
# NOTE: the hash is the file name excluding any file extensions
# Returns the matching file, its type (raspbian or armhf), and its package name

path = " " # path to armhf.json and raspbian.json files


x = 0
# open the json file as read-only
with open(path + "raspbian.json", "r") as read_file:
    rasp_data = json.load(read_file)
# change here 1
with open(path + "armhf.json", "r") as hf_read_file:
    armhf_data = json.load(hf_read_file)

def match(hash):
    global x
    for i in rasp_data:
        rasp_hash = i["filehash"]
        if hash == rasp_hash:
            rasp = i["filename"]
            rasp = rasp.split("/")[-1]
            rasp_db = i["deb_package"]
            rasp_db = rasp_db.split("_")[0:-1]
            for v in armhf_data:
                arm = v["filename"]
                arm = arm.split("/")[-1]
                arm_db = v["deb_package"]
                arm_db = arm_db.split("_")[0:-1]
                if rasp == arm and rasp_db == arm_db:
                    print("1) rasp: ", rasp, i["filehash"])
                    print("2) armhf: ", arm, v["filehash"])
                    x = x + 1
                else:
                    x = x
    if x == 0:
        for v in armhf_data:
            arm_hash = v["filehash"]
            if hash == arm_hash:
                arm = v["filename"]
                arm = arm.split("/")[-1]
                arm_db = v["deb_package"]
                arm_db = arm_db.split("_")[0:-1]
                for i in rasp_data:
                    rasp = i["filename"]
                    rasp = rasp.split("/")[-1]
                    rasp_db = i["deb_package"]
                    rasp_db = rasp_db.split("_")[0:-1]
                    if rasp == arm and rasp_db == arm_db:
                        print("1) rasp: ", rasp, i["filehash"])
                        print("2) armhf: ", arm, v["filehash"])
                        x = x + 1
                    else:
                        x = x

    print(x, " matches found!")


filehash = input("Enter the file hash: ")

match(filehash)

