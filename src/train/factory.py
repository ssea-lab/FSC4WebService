import train.maml as maml
import train.regular as regular
import train.finetune as finetune


def train(train_data, val_data, model, args):
   if args.maml:
        return maml.train(train_data, val_data, model, args)
   else:
        return regular.train(train_data, val_data, model, args)


def test(test_data, model, args, num_episodes, return_score=False, vocab=None, verbose=True):

    if args.maml:
        return maml.test(test_data, model, args, num_episodes)
    elif args.mode == 'finetune':
        return finetune.test(test_data, model, args, num_episodes)
    else:
        return regular.test(test_data, model, args, num_episodes, return_score=return_score, vocab=vocab)
