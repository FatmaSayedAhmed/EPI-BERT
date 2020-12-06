def nkmer(inFile, outFile, nkmer):
    output_lines = []
    with open(inFile) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] != '\n':
                tempLine = line.replace(" ", "")
                length = len(tempLine)
                str = ""
                for i in range(0, length):
                    if (i % nkmer == 0 and i != 0):
                        str += " "
                    str += tempLine[i]
                str += "\n"
                output_lines.append(str)
    f.close()

    with open(outFile, "w") as f:
        for line in output_lines:
            f.write(line)
    f.close()


nkmer("pre_train_data_128/EP_pre_train_data_128_1kmer.txt", "pre_train_data_128/EP_pre_train_data_128_2kmer.txt",2)
