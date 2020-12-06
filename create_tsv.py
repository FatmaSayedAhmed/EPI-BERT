# coding:utf-8

# coding:utf-8

def create_tsv(pos_f, neg_f, out_f, gap_length):
    out_lines = []
    with open(pos_f) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] != ">":
                line = line.lower()
                length = len(line.strip())
                seq = ""
                for i in range(gap_length, length, gap_length):
                    seq += line[i - gap_length: i] + " "
                seq += line[i:]
                out_lines.append("train\t1\t\t" + seq)

    with open(neg_f) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] != ">":
                line = line.lower()
                length = len(line.strip())
                seq = ""
                for i in range(gap_length, length, gap_length):
                    seq += line[i - gap_length: i] + " "
                seq += line[i:]
                out_lines.append("train\t0\t\t" + seq)

    with open(out_f, "w") as f:
        for line in out_lines:
            f.write(line)

kmer = 4
for cell_line in ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']:
    for t in ["enhancer", "promoter"]:
        for s in ["te", "tr"]:
    #        for kmer in range(1, 4):
            create_tsv("Dataset/asFASTA/Pos_Neg_data/" + cell_line + "/" + cell_line + "_" + t + "_pos_" + s + ".fasta",
                       "Dataset/asFASTA/Pos_Neg_data/" + cell_line + "/" + cell_line + "_" + t + "_neg_" + s + ".fasta",
                       "Dataset/asTSV/" + str(kmer) + "kmer_tsv_data/" + cell_line + "/" + cell_line + "_" + t + "_sorted_" + s + ".tsv", kmer)



# All Data for general EPI-BERT model
# kmer = 4
# for t in ["enhancer", "promoter"]:
#     create_tsv("DatasetAll/asFASTA/" + t + "_All_pos_tr.fasta",
#                "DatasetAll/asFASTA/" + t + "_All_neg_tr.fasta",
#                "DatasetAll/asTSV/" + str(kmer) + "kmer_tsv_data/" + t + "_All_sorted_tr.tsv", kmer)
