# KGen

python train.py --datapath ../data --dataset WNUT17 --base_model roberta --train_text few_shot_20 --train_ner few_shot_20 --train_ent few_shot_20 --few_shot_sets 10 --test_text test.words --test_ner test.ner --test_ent test.ent --epoch 200 --max_seq_len 128 --batch_size 32 --lr 1e-05 --use_truecase False --unsup_lr 0.5 --unsup_text train_0.words --unsup_ner train_0.ner --model_name ../results/wnut17_naiveft_roberta_seq128_epoch --load_dataset False --use_gpu True --load_model True --load_model_name ./pretrained_models/ontonotes5_naiveft_roberta_seq128_epoch4.pt
