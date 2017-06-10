import argparse


def add_and_parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hidden_layers',
                        type=int,
                        required=True,
                        help='The number of hidden layers')
    parser.add_argument('--num_neurons',
                        type=int,
                        help='The number of neurons per layer')
    parser.add_argument('--learning_rate',
                        type=float,
                        help='The learning rate')
    parser.add_argument('--regularization',
                        type=float,
                        help='regularization parameter')
    parser.add_argument('--predict',
                        action='store_true',
                        help='Flag to predict')
    parser.add_argument('--skip_validation',
                        action='store_true',
                        default=False,
                        help='Does not exit if validation is getting worse. \
                        Use this only when training the model for short \
                        training epochs if you need to quickly tune hyper \
                        parameters. Note this mode DOES NOT SAVE the model. \
                        Hence use it only for short training cycles.')

    parser.parse_args()
    args, unknown = parser.parse_known_args()

    if args.predict is False:
        if args.regularization is None:
            parser.error("Regularization is required when not predicting.")
        if args.learning_rate is None:
            parser.error("Learning rate is required when not predicting.")

    if args.learning_rate <= 0 or args.learning_rate >= 1:
        parser.error("Learning rate must be greater than 0 and less than 1.")

    if args.regularization < 0:
        parser.error("Regularization must be greater than or equal to 0.")

    if args.num_hidden_layers != 0 and args.num_neurons is None:
        parser.error("Num_neurons is required if num_hidden_layers > 0.")

    return args, unknown
