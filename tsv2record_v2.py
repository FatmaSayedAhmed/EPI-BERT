# coding:utf-8

from run_classifier import ColaProcessor
import tokenization
from run_classifier import file_based_convert_examples_to_features
import random

def create_tfrecord(kmer, cell_line):

    tsv_root = "Dataset/asTSV/" + str(kmer) + "kmer_tsv_data/" + cell_line
    tfrecord_root = "Dataset/asTF_Record/" + str(kmer) + "kmer_tfrecord/" + cell_line

    vocab_file = "vocab/vocab_" + str(kmer) + "kmer.txt"
    processor = ColaProcessor()
    label_list = processor.get_labels()
    examples = processor.fatma_get_dev_examples(tsv_root + "/" , cell_line)
    train_file = tfrecord_root + "/" + cell_line + "_dev.tf_record"
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=True)

    file_based_convert_examples_to_features(
            examples, label_list, 128, tokenizer, train_file)

    examples = processor.fatma_get_train_examples(tsv_root + "/", cell_line)
    train_file = tfrecord_root + "/" + cell_line + "_train.tf_record"
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
            examples, label_list, 128, tokenizer, train_file)



def create_tfrecord_All(kmer):

    tsv_root = "DatasetAll/asTSV/" + str(kmer) + "kmer_tsv_data/"
    tfrecord_root = "DatasetAll/asTF_Record/" + str(kmer) + "kmer_tfrecord/"

    vocab_file = "vocab/vocab_" + str(kmer) + "kmer.txt"
    processor = ColaProcessor()
    label_list = processor.get_labels()

    examples = processor.fatma_get_train_examples_All(tsv_root)
    train_file = tfrecord_root + "train_All.tf_record"
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
            examples, label_list, 128, tokenizer, train_file)

#for kmer in range(1, 4):
kmer = 4
create_tfrecord_All(kmer)

# for cell_line in ['GM12878', 'HeLa-S3', 'HUVEC', 'IMR90', 'K562', 'NHEK']:
#     create_tfrecord(kmer, cell_line)
