import random
def MergSorted_SplitUnsorted_EP_Pairs(en_sorted_file, pr_sorted_file, en_file, pr_file):
    merged_en_pr_samples = []
    with open(en_sorted_file,"r") as f:
        en_samples = f.readlines()
    f.close()

    with open(pr_sorted_file,"r") as f:
        pr_samples = f.readlines()
    f.close()

    for (en_line, pro_line) in zip(en_samples,pr_samples):
        merged_en_pr_line = en_line + "," + pro_line
        merged_en_pr_samples.append(merged_en_pr_line)

    random.shuffle(merged_en_pr_samples)
    en_samples.clear()
    pr_samples.clear()
    for merged_en_pr_line in merged_en_pr_samples:
        line = str(merged_en_pr_line)
        temp =line.partition(",")
        en_samples.append(temp[0])
        pr_samples.append(temp[2])


    with open(en_file,"w") as f:
        for sample in en_samples:
            f.write(sample)
        f.close()

    with open(pr_file,"w") as f:
        for sample in pr_samples:
            f.write(sample)
        f.close()


kmer = 4
for cell_line in ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']:
    for s in ["te", "tr"]:
    #        for kmer in range(1, 4):
        MergSorted_SplitUnsorted_EP_Pairs("Dataset/asTSV/" + str(kmer) + "kmer_tsv_data/" + cell_line + "/" + cell_line + "_enhancer_sorted_" + s + ".tsv",
                   "Dataset/asTSV/" + str(kmer) + "kmer_tsv_data/" + cell_line + "/" + cell_line + "_promoter_sorted_" + s + ".tsv",
                   "Dataset/asTSV/" + str(kmer) + "kmer_tsv_data/" + cell_line + "/" + cell_line + "_enhancer_" + s + ".tsv",
                   "Dataset/asTSV/" + str(kmer) + "kmer_tsv_data/" + cell_line + "/" + cell_line + "_promoter_" + s + ".tsv",)



# kmer = 4
# MergSorted_SplitUnsorted_EP_Pairs("DatasetAll/asTSV/" + str(kmer) + "kmer_tsv_data/" + "enhancer_All_sorted_tr.tsv",
#            "DatasetAll/asTSV/" + str(kmer) + "kmer_tsv_data/" + "promoter_All_sorted_tr.tsv",
#            "DatasetAll/asTSV/" + str(kmer) + "kmer_tsv_data/" + "enhancer_All_tr.tsv",
#            "DatasetAll/asTSV/" + str(kmer) + "kmer_tsv_data/" + "promoter_All_tr.tsv")
#
