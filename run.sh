cuda=0
way=10
shot=1
query=30
embedding=meta  #[cnn, lstmatt, meta, bert]
classifier=r2d2 #[routing, proto, gnn, r2d2]
feature_convergence=sum #[sum,max,mean,cnn]
val_episodes=100
patience=20
seed=2021
aug_support=1
meta_label_sim=1

dataset=pw #[pw,aws]
data_path="data/"$dataset".json"
wv_vector=$dataset"_word2vec.txt"
index2vector_path="data/"$dataset"_index2vector.pkl"

if [ $dataset = "pw" ] ; then
    n_train_class=145
    n_val_class=60
    n_test_class=59
elif [ $dataset = "aws" ] ; then
    n_train_class=51
    n_val_class=23
    n_test_class=22
else
    echo "Invalid dataset!"
    exit 1
fi

cmd="python src/main.py 
    --cuda $cuda 
    --way $way 
    --shot $shot 
    --query $query 
    --embedding $embedding
    --classifier $classifier 
    --dataset $dataset 
    --data_path $data_path 
    --n_train_class $n_train_class 
    --n_val_class $n_val_class 
    --n_test_class $n_test_class 
    --word_vector $wv_vector
    --val_episodes $val_episodes 
    --patience $patience  
    --seed $seed 
    --feature_convergence $feature_convergence 
    --index2vector_path $index2vector_path
    --meta_iwf 
    "

if [ $aug_support ] ; then
    cmd=$cmd"--aug_support "
fi

if [ $meta_label_sim ] ; then
    cmd=$cmd"--meta_label_sim "
fi

eval $cmd