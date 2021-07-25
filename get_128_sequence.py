def get_128_sequence(inFile, outFile):
    outSamples =[]
    with open(inFile) as f:
        lines = f.readlines()
        for line in lines:
            subline = line[:256]
            outSamples.append(subline)
            outSamples.append('\n')

    with open(outFile, "w") as f:
        for line in outSamples:
            f.write(line)


get_128_sequence("pre_train_data_128/EP_pre_train_data.txt",
                 "pre_train_data_128/EP_pre_train_data_128_1kmer.txt")

################################################################################################

# def get_128_sequence(inFile, outFile):
#     outSamples =[]
#     with open(inFile) as f:
#         lines = f.readlines()
#         for line in lines:
#             if line[0] != '>':
#                 subline = line[:128]+'\n'
#                 outSamples.append(subline)
#             else:
#                 outSamples.append(line)
#
#     with open(outFile, "w") as f:
#         for line in outSamples:
#             f.write(line)
#
# # cell_line = "IMR90"
# # for cell_line in ['HUVEC', 'K562', 'NHEK']:
# for cell_line in ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90' , 'K562', 'NHEK']:
#     for s1 in ["enhancer", "promoter"]:
#         for s2 in ["neg", "pos"]:
#             for s3 in ["te", "tr"]:
#                 get_128_sequence("Dataset_V5/asFASTA/Pos_Neg_data/"  + cell_line + "/" + cell_line + "_" + s1 + "_" + s2 + "_" + s3 + ".fasta",
#                                  "Dataset/asFASTA/Pos_Neg_data/"  + cell_line + "/" + cell_line + "_" + s1 + "_" + s2 + "_" + s3 + ".fasta")
