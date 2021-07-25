# 设置可以使用的GPU
# export CUDA_VISIBLE_DEVICES=1
python run_classifier_All.py \
  --train_data_name=All \
  --test_data_name=GM12878  \
  --train_data_root=./DatasetAll/asTF_Record/1kmer_tfrecord/ \
  --test_data_root=./Dataset/asTF_Record/1kmer_tfrecord/ \
  --do_eval=True \
  --num_train_epochs=100 \
  --batch_size=16 \
  --bert_config=./bert_config_1.json \
  --vocab_file=./vocab/vocab_1kmer.txt \
  --init_checkpoint=./model/1kmer_model_128/model.ckpt \
  --save_path=./model/1kmer_Classifier_model_128_All/model_EPI_General.ckpt
