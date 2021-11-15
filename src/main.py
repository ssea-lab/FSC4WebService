import os
import pickle
import argparse
import torch
import random
import numpy as np
import embedding.factory as ebd
import classifier.factory as clf
import dataset.loader as loader
import train.factory as train_utils
import logging
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot Text Classification with Distributional Signatures")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="data/pw.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="pw",
                        help="name of the dataset. "
                        "Options: [pw, aws]")
    parser.add_argument("--n_train_class", type=int, default=146,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=60,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=59,
                        help="number of meta-test classes")

    # load bert embeddings for sent-level datasets (optional)
    parser.add_argument("--n_workers", type=int, default=10,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")
    parser.add_argument("--bert", default=False, action="store_true",
                        help=("set true if use bert embeddings "
                              "(only available for sent-level datasets: "
                              "huffpost, fewrel"))
    parser.add_argument("--bert_cache_dir", default='cache', type=str,
                        help=("path to the cache_dir of transformers"))
    parser.add_argument("--pretrained_bert", default=None, type=str,
                        help=("path to the pre-trained bert embeddings."))

    # task configuration
    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=5,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")

    # train/test configuration
    parser.add_argument("--train_epochs", type=int, default=1000,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    # model options
    parser.add_argument("--embedding", type=str, default="avg",
                        help=("document embedding method. Options: "
                              "[avg, idf, iwf, meta, cnn]"))
    parser.add_argument("--classifier", type=str, default="nn",
                        help=("classifier. Options: [nn, proto, r2d2, mlp, gnn]"))

    # cnn configuration
    parser.add_argument("--cnn_num_filters", type=int, default=50,
                        help="Num of filters per filter size [default: 50]")
    parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
                        default=[3, 4, 5],
                        help="Filter sizes [default: 3]")

    # nn configuration
    parser.add_argument("--nn_distance", type=str, default="l2",
                        help=("distance for nearest neighbour. "
                              "Options: l2, cos [default: l2]"))

    # proto configuration
    parser.add_argument("--proto_hidden", nargs="+", type=int,
                        default=[300, 300],
                        help=("hidden dimension of the proto-net"))

    # maml configuration
    parser.add_argument("--maml", action="store_true", default=False,
                        help=("Use maml or not. "
                        "Note: maml has to be used with classifier=mlp"))
    parser.add_argument("--mlp_hidden", nargs="+", type=int, default=[300, 5],
                        help=("hidden dimension of the proto-net"))
    parser.add_argument("--maml_innersteps", type=int, default=10)
    parser.add_argument("--maml_batchsize", type=int, default=10)
    parser.add_argument("--maml_stepsize", type=float, default=1e-1)
    parser.add_argument("--maml_firstorder", action="store_true", default=False,
                        help="truncate higher order gradient")

    # lrd2 configuration
    parser.add_argument("--lrd2_num_iters", type=int, default=5,
                        help=("num of Newton steps for LRD2"))

    # induction networks configuration
    parser.add_argument("--induct_rnn_dim", type=int, default=128,
                        help=("Uni LSTM dim of induction network's encoder"))
    parser.add_argument("--induct_hidden_dim", type=int, default=100,
                        help=("tensor layer dim of induction network's relation"))
    parser.add_argument("--induct_iter", type=int, default=3,
                        help=("num of routings"))
    parser.add_argument("--induct_att_dim", type=int, default=64,
                        help=("attention projection dim of induction network"))

    # base word embedding
    parser.add_argument("--wv_path", type=str,
                        default="./cache/",
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default="wiki.en.vec",
                        help=("Name of pretrained word embeddings."))

    # options for the distributional signatures
    parser.add_argument("--meta_idf", action="store_true", default=False,
                        help="use idf")
    parser.add_argument("--meta_iwf", action="store_true", default=False,
                        help="use iwf")
    parser.add_argument("--meta_label_sim", action="store_true", default=False,
                        help="use label sim")
    parser.add_argument("--meta_task_idf", action="store_true", default=False,
                        help="use task idf")
    parser.add_argument("--meta_w_target", action="store_true", default=False,
                        help="use target importance score")
    parser.add_argument("--meta_w_target_lam", type=float, default=1,
                        help="lambda for computing w_target")
    parser.add_argument("--meta_target_entropy", action="store_true", default=False,
                        help="use inverse entropy to model task-specific importance")
    parser.add_argument("--meta_ebd", action="store_true", default=False,
                        help="use word embedding into the meta model "
                        "(showing that revealing word identity harm performance)")
    parser.add_argument("--aug_support", action="store_true", default=False,
                        help="aug support")
    parser.add_argument("--feature_convergence", type=str, default="sum",
                        help=("feature convergence function"
                              "[sum, max, mean, cnn]"))
    
    # training options
    parser.add_argument("--seed", type=int, default=330, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--patience", type=int, default=20, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="train",
                        help=("Running mode."
                              "Options: [train, test]"
                              "[Default: train]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")
    parser.add_argument("--iatt", action="store_true", default=False,
                        help="use instance attention for prototype net")
    
    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if args.embedding != "cnn" and attr[:4] == "cnn_":
            continue
        if args.classifier != "proto" and attr[:6] == "proto_":
            continue
        if args.classifier != "nn" and attr[:3] == "nn_":
            continue
        if args.embedding != "meta" and attr[:5] == "meta_":
            continue
        if args.embedding != "cnn" and attr[:4] == "cnn_":
            continue
        if args.classifier != "mlp" and attr[:4] == "mlp_":
            continue
        if args.classifier != "proto" and attr[:6] == "proto_":
            continue
        if "pos" not in args.auxiliary and attr[:4] == "pos_":
            continue
        if not args.maml and attr[:5] == "maml_":
            continue
        print("\t{}={}".format(attr.upper(), value))
    
def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    print_args(args)
    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    # initialize model
    model = {}
    model["ebd"] = ebd.get_embedding(vocab, args)
    model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)
    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train_utils.train(train_data, val_data, model, args)

    # testing on validation data
    val_acc, val_std = train_utils.test(val_data, model, args,
                                        args.val_episodes)
    
    # testing on test data
    test_acc, test_std = train_utils.test(test_data, model, args,
                                          args.test_episodes, return_score=True, vocab=vocab)
    print("Test acc: {:.4f} ± {:.4f}".format(test_acc, test_std),flush=True)
    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
            "val_acc": val_acc,
            "val_std": val_std
        }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
