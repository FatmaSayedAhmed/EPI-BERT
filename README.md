# EPI-BERT

# Introduction
This is a model for identification Enhancer Promoter Interaction (EPI) based on BERT which is proposed.
We pretrained a BERT model through amount of enhancer and promoter sequences (see the paper for more details).
We train the model on 6 different cell lines datasets and evaluate its performance.
We merge all the datasets and train a model.
# How to Start
You should first clone the project by command
>git clone https://github.com/FatmaSayedAhmed/EPI-BERT.git

Then you need to download models and datasets from the address:
>https://drive.google.com/drive/folders/10NfYgYYrVqXyQHhJnCzKl_vCb3UXqQal?usp=sharing
 
Then you should uzip these zips and put them on the root of the project.

# Requirments 
python                    3.7.6

tensorflow                1.15.0

tensorflow-gpu            2.1.0

six                       1.13.0

numpy                     1.17.4

scikit-learn              0.22

# Pre-training
 
 You should create data for pre-train by the command
 >sh create_data.sh
 
You should ensure the content of file pre_train.sh
>input_file is your input data for pre-training whose format is tf_record.  
output_dir is the dir of your output model.  
bert_config_file defines the structure of the model.
train_batch_size should be change more little if your computer don't support so big batch size.
You can change the num_train_steps by yourself.

After ensuring the content, then you can pre-trian your model by the command:
>sh pre_train.sh

 # Fine-Tuning & Evaluation & Save Model
 First you should prepare your data that will be used for fine-tuning by runing all of the following commands in order
> python create_tsv.py 
  
> python MergSorted_SplitUnsorted_EP_Pairs.py
  
> python tsv2record_v2.py
 
 When you ready to fine-tune the model or do other, you should open one of these files first:
 file run_fine_tune.sh ---> for fine-tune using specific cell line,
 file run_fine_tune_All.sh ---> for fine-tune using All cell lines,
 file run_fine_tune_eval_only.sh ---> for evaluate only using specific cell line,
 Then you should change the parameters according to your needs.
> do_eval and do_save are used to indicate if you want to evaluate the model or save the final model.  
If the do_save is True then the final model will be saved in path "./model/(k)mer_classifier_model_128/".
train_dict and test_dict record the numbers of samples in training sets and test sets.  
init_chechpoint is the model which is used to train.

After ensuring the content, then you can fine-tune your model by one of these commands:
> sh run_fine_tune.sh
 
> sh run_fine_tune_All.sh
 
> sh run_fine_tune_eval_only.sh

# Predict
You can predict your enhancer-promoter pairs data by command
> python predict_EPI.py f1 f2 f3  


f1 and f2 are the fasta format files contain your enhancer and promoter sequneces pairs respectively, and f3 is the output file.

