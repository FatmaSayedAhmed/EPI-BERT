lines = []
vocab = []
vocab.append('a')
vocab.append('g')
vocab.append('c')
vocab.append('t')
line =''

for v1 in vocab:
    line = v1 + '\n'
    lines.append(line)

with open("vocab_1kmer.txt","w") as f:
    for line in lines:
        f.write(line)
    f.write("[CLS]\n")
    f.write("[SEP]\n")
    f.write("[UNK]\n")
    f.write("[MASK]\n")
    f.close()
############################################
for v1 in vocab:
    line = v1
    for v2 in vocab:
        line += v2
        line +='\n'
        lines.append(line)
        line = v1

with open("vocab_2kmer.txt","w") as f2:
    for line in lines:
        f2.write(line)
    f2.write("[CLS]\n")
    f2.write("[SEP]\n")
    f2.write("[UNK]\n")
    f2.write("[MASK]\n")
    f2.close()
###################################################

for v1 in vocab:
    line = v1
    for v2 in vocab:
        line += v2
        for v3 in vocab:
            line += v3
            line +='\n'
            lines.append(line)
            line = v1 + v2
        line = v1

with open("vocab_3kmer.txt","w") as f3:
    for line in lines:
        f3.write(line)
    f3.write("[CLS]\n")
    f3.write("[SEP]\n")
    f3.write("[UNK]\n")
    f3.write("[MASK]\n")
    f3.close()
############################################################################

for v1 in vocab:
    line = v1
    for v2 in vocab:
        line += v2
        for v3 in vocab:
            line += v3
            for v4 in vocab:
                line += v4
                line +='\n'
                lines.append(line)
                line = v1 + v2 + v3
            line = v1 + v2
        line = v1

with open("vocab_4kmer.txt","w") as f4:
    for line in lines:
        f4.write(line)
    f4.write("[CLS]\n")
    f4.write("[SEP]\n")
    f4.write("[UNK]\n")
    f4.write("[MASK]\n")
    f4.close()
############################################################################
