import argparse
import utils
import torch


def get_args():
    parser = argparse.ArgumentParser("training hyperparameters")

    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        metavar='E',
        help="number of epochs (default: 2)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=60,
        metavar='B',
        help="batch size for training (default: 60)"
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=300,
        metavar='TB',
        help="batch size for inference (default: 300)"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help="batch size for training (default: 0.001)"
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        metavar='B1',
        help="coefficients used for computing running averages of gradient in Adam's optimizer (default: 0.9)"
    )
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.99,
        metavar='B2',
        help="coefficients used for computing running averages of the square of the gradient in Adam's optimizer (default: 0.99)"
    )
    parser.add_argument(
        '--use_cuda',
        action='store_false',
        default=False,
        help="whether to train on cuda"
    )
    parser.add_argument(
        '--model_dir',
        default=None,
        metavar='MD',
        help="direcory to save the model (default: None)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=7,
        metavar='S',
        help="seeding (default: 7)"
    )
    args = parser.parse_args()
    return args


def test(model, test_loader, loss_func, device):
    running_loss = 0.0
    correct_preds = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (_, features, labels) in enumerate(test_loader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(predictions == labels).item()
            running_loss += loss_func(outputs, labels).item()
    loss = running_loss/batch_idx
    accuracy = 100.0 * correct_preds/len(test_loader.dataset)
    return loss, accuracy


def train(model, train_loader, test_loader, epochs, loss_func, optimizer, device):
    best_model_wts = model.state_dict()
    best_test_accuracy = 0.0
    best_epoch = 0

    train_running_loss = 0.0
    train_correct_preds = 0
    for epoch in range(epochs):
        for batch_idx, (_, features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # metrics
            predictions = torch.argmax(outputs, dim=1)
            train_correct_preds += torch.sum(predictions == labels).item()
            train_running_loss += loss_func(outputs, labels).item()

        # logging
        train_loss = train_running_loss/batch_idx
        train_accuracy = 100.0 * train_correct_preds/len(train_loader.dataset)
        test_loss, test_accuracy = test(model, test_loader, loss_func, device)
        print("*******Epoch: [{}/{}]*******\nTrain loss: {:.6f}\tTrain accuracy: {:.2f}\nTest loss: {:.6f}\tTest accuracy: {:.2f}".format(
               epoch, epochs, train_loss, train_accuracy, test_loss, test_accuracy))

        # saving best model
        if test_accuracy > best_test_accuracy:
            best_model_wts = model.state_dict()
            best_epoch = epoch
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy

        performance_metrics = {'optimal_epoch': best_epoch,
                               'train_accuracy': best_train_accuracy,
                               'test_accuracy': best_test_accuracy}
    return model.load_state_dict(best_model_wts), performance_metrics


if __name__ == "__main__":
    args = get_args()

    if args.use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(args.seed)

    train_set = utils.ImageDataset(utils.train_dir, utils.train_labels_df, True)
    train_loader = utils.ImageLoader(train_set, args.batch_size, True)

    test_set = utils.ImageDataset(utils.test_dir, utils.test_labels_df, True)
    test_loader = utils.ImageLoader(test_set, args.test_batch_size, False)

    model = utils.Model().to(device)
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    model, metrics = train(model, train_loader, test_loader, args.epochs, loss_func, optimizer, device)
