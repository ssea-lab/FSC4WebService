cuda=0
way=5
shot=1
query=30
embedding=meta
classifier=r2d2
feature_convergence = sum
val_episodes=100
patience=20
seed=2021
aug_support = 1

dataset=pw # [pw,aws]
data_path="data/"$dataset".json"
wv_vector=$dataset"_word2vec.txt"

if [$dataset="pw"]; then
    n_train_class=146
    n_val_class=60
    n_test_class=59
elif [$dataset="aws"];then
    n_train_class=51
    n_val_class=23
    n_test_class=22
else
    echo "Invalid dataset!"
    exit 1

cmd=python src/main.py \
    --cuda $cuda \
    --way $way \
    --shot $shot \
    --query $query \
    --embedding $embedding  \
    --classifier $classifier \
    --dataset $dataset \
    --data_path $data_path \
    --n_train_class $n_train_class \
    --n_val_class $n_val_class \
    --n_test_class $n_test_class \
    --word_vector $wv_vector \
    --val_episodes $val_episodes \
    --patience $patience  \
    --seed $seed \
    --feature_convergence $feature_convergence \

if [$aug_support]; then
    cmd="$cmd--aug_support \"
    
eval $cmd