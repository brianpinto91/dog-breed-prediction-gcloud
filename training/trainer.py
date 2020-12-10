import argparse
import utils
import torch
import pandas as pd

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
        '--log_interval',
        type=int,
        default=10,
        metavar="L",
        help="logging interval i.e the number of minibatches after which the metrics are logged"
    )
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        default=False,
        help="whether to train on cuda"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=utils.DATA_DIR,
        metavar='D',
        help="location of the directory containing train.h5 and test.h5 files"
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default=utils.MODEL_DIR,
        metavar='MD',
        help="direcory to save the model (default: None)"
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=utils.LOG_DIR,
        metavar='LOG',
        help="location of the directory where the training log is to be saved"
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
        for batch_idx, (features, labels) in enumerate(test_loader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(predictions==labels).item()
            running_loss += loss_func(outputs, labels).item()
    loss = running_loss / (batch_idx + 1)
    accuracy = 100.0 * correct_preds / len(test_loader.dataset)
    return loss, accuracy


def train(args, model, train_loader, test_loader, loss_func, optimizer, device):
    best_model_wts = model.state_dict()
    best_test_accuracy = 0.0
    best_epoch = 0
    log_list = []
    for epoch in range(args.epochs):
        train_running_loss = 0.0
        train_correct_preds = 0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)
            train_correct_preds += torch.sum(predictions==labels).item()
            train_running_loss += loss.item()
            if (batch_idx % args.log_interval == 0):
              print("Epoch {}:\tBatch: [{}/{}]:\tRunning avg loss: {:.6f}".format(epoch + 1,
                    batch_idx, int(len(train_loader.dataset)/train_loader.batch_size),
                    train_running_loss / (batch_idx+1)), flush=True)
        #logging
        train_loss = train_running_loss / (batch_idx + 1) 
        train_accuracy = 100.0 * train_correct_preds / len(train_loader.dataset)
        test_loss, test_accuracy = test(model, test_loader, loss_func, device)
        log_list.append([epoch + 1, train_loss, test_loss, train_accuracy, test_accuracy])
        print("*******Epoch: [{}/{}]*******\nTrain loss: {:.6f}\tTrain accuracy: {:.2f}\nTest loss: {:.6f}\tTest accuracy: {:.2f}".format(
               epoch + 1, args.epochs, train_loss, train_accuracy, test_loss, test_accuracy), flush=True)
        #saving best model
        if test_accuracy > best_test_accuracy:
            best_model_wts = model.state_dict()
            best_epoch = epoch + 1 # epoch starts from zero
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy
        
        best_performance_metrics = {'optimal_epoch': best_epoch,
                                'train_accuracy': best_train_accuracy,
                                 'test_accuracy': best_test_accuracy}
    model.load_state_dict(best_model_wts)
    log_df = pd.DataFrame(log_list, 
                          columns=['epoch', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'])
    return model, best_performance_metrics, log_df


if __name__ == "__main__":
    args = get_args()

    if args.use_cuda:
        try:
            device = torch.device('cuda')
        except Exception as e:
            print("Check if you have correct nvidia driver installed!")
            print(e)
            device = None # lets throw error when loading model on device. Do not let run on cpu with gpu resources on cloud!
    else:
        device = torch.device('cpu')

    torch.manual_seed(args.seed)

    model = utils.Model().to(device)

    train_file_path, test_file_path = utils.get_data_paths(args)
    train_set = utils.ImageDataset(train_file_path, aug_images=True)
    train_loader = utils.image_loader(train_set, args.batch_size, True)
    test_set = utils.ImageDataset(test_file_path, aug_images=False)
    test_loader = utils.image_loader(test_set, args.test_batch_size, False)

    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    model, best_performance_metrics, log_df = train(args, model, train_loader, test_loader, loss_func, optimizer, device)
    utils.save_model(args, model)
    utils.save_job_log(args, log_df, best_performance_metrics)