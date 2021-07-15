MODEL=$1
DEVICE=$2

echo "Start benchmarking $MODEL with embedding size 200"
CUDA_VISIBLE_DEVICES=$DEVICE DGLBACKEND=pytorch dglke_train --model_name $MODEL \
--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 200 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
--batch_size_eval 128 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test > "${MODEL}_dim200.log"

echo "Start benchmarking $MODEL with embedding size 300"
CUDA_VISIBLE_DEVICES=$DEVICE DGLBACKEND=pytorch dglke_train --model_name $MODEL \
--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 300 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
--batch_size_eval 128 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test > "${MODEL}_dim300.log"

#echo "Start benchmarking $MODEL with embedding size 600"
#CUDA_VISIBLE_DEVICES=$DEVICE DGLBACKEND=pytorch dglke_train --model_name $MODEL \
#--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
#--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 600 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
#--batch_size_eval 128 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test > "${MODEL}_dim600.log"
#
#echo "Start benchmarking $MODEL with embedding size 800"
#CUDA_VISIBLE_DEVICES=$DEVICE DGLBACKEND=pytorch dglke_train --model_name $MODEL \
#--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
#--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 800 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
#--batch_size_eval 128 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test > "${MODEL}_dim800.log"
#
#echo "Start benchmarking $MODEL with embedding size 1000"
#CUDA_VISIBLE_DEVICES=$DEVICE DGLBACKEND=pytorch dglke_train --model_name $MODEL \
#--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
#--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 1000 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
#--batch_size_eval 128 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test > "${MODEL}_dim1000.log"

