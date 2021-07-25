def create_pre_train_data(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, output_file):
    output_lines = []
    for file in {f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12}:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                if line[0] != ">":
                    length = len(line.strip())
                    str = ""
                    for i in range(0,length):
                        str += line[i] + " "
                    str += "\n"
                    output_lines.append(str)
        f.close()

    with open(output_file,"w") as f:
        for line in output_lines:
            f.write(line)
            f.write("\n")
    f.close()

create_pre_train_data("./Dataset/asFASTA/Original_Data/GM12878/GM12878_enhancers.fasta",
                      "./Dataset/asFASTA/Original_Data/GM12878/GM12878_promoters.fasta",
                      "./Dataset/asFASTA/Original_Data/HeLa-S3/HeLa-S3_enhancers.fasta",
                      "./Dataset/asFASTA/Original_Data/HeLa-S3/HeLa-S3_promoters.fasta",
                      "./Dataset/asFASTA/Original_Data/HUVEC/HUVEC_enhancers.fasta",
                      "./Dataset/asFASTA/Original_Data/HUVEC/HUVEC_promoters.fasta",
                      "./Dataset/asFASTA/Original_Data/IMR90/IMR90_enhancers.fasta",
                      "./Dataset/asFASTA/Original_Data/IMR90/IMR90_promoters.fasta",
                      "./Dataset/asFASTA/Original_Data/K562/K562_enhancers.fasta",
                      "./Dataset/asFASTA/Original_Data/K562/K562_promoters.fasta",
                      "./Dataset/asFASTA/Original_Data/NHEK/NHEK_enhancers.fasta",
                      "./Dataset/asFASTA/Original_Data/NHEK/NHEK_promoters.fasta",
                      "./pre_train_data_128/EP_pre_train_data.txt")

