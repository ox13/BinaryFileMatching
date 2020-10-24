import itertools

word_lengths = [12,4,2,1]
stringi = ""
vocab_file = open("/Users/kenlampinen/Desktop/data/armhf_vocab.txt", "w")

for n in word_lengths:
    lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
    for j in lst:
        for i in j:
            stringi += str(i)
        stringi += "\n"

print(stringi)

vocab_file.write(stringi)
vocab_file.close()
