# 设置可以使用的GPU
# export CUDA_VISIBLE_DEVICES=1
python run_classifier_V2.py \
  --data_name=GM12878 \
  --data_root=./Dataset/asTF_Record/1kmer_tfrecord/GM12878/ \
  --do_eval=True \
  --num_train_epochs=100 \
  --batch_size=16 \
  --bert_config=./bert_config_1.json \
  --vocab_file=./vocab/vocab_1kmer.txt \
  --init_checkpoint=./model/1kmer_model_128/model.ckpt \
  --save_path=./model/1kmer_Classifier_model_128/model_EPI_GM12878_100.ckpt
