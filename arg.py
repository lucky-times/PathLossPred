import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    dir = '/home/root888/LC_low_attitude/dataset_0723'
    # Required parameters
    parser.add_argument("--output_dir", default='./result', type=str, help="The output directory where the model checkpoints and predictions will be written.",
    )
    
    # Other parameters
    parser.add_argument("--train_file", default=dir+'/train', type=str, help="The input training file.")
    parser.add_argument("--dev_file", default=dir+'/test', type=str, help="The input evaluation file.")
    parser.add_argument("--test_file", default=dir+'/test', type=str, help="The input testing file.")

    parser.add_argument("--learning_rate", default=0.00001, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_epochs_per_decay", default=2, type=int, help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_decay_factor", default=0.95, type=float,help="Weight decay if we apply some.")
    parser.add_argument("--num_train_epochs", default=100, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=16, type=int)

    parser.add_argument("--output_numclass", default=1, type=int)
    parser.add_argument("--map_size", default=512, type=int)

    parser.add_argument("--do_train", action="store_false", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to save predicted labels.")


    args = parser.parse_args()
    return args