import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from models import CLDNN
from dataloader import FederatedDataset
from train_utils import train_model, test
from aggregation import fed_avg, per_layer_aggregation
from temporal_averaging import temporal_avg


def parse_args():
    parser = argparse.ArgumentParser(description='TabFIDS: Federated Intrusion Detection')
    parser.add_argument('--n_devices', type=int, default=11)
    parser.add_argument('--n_rounds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--output_path', type=str, default='output/')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--per_layer', action='store_true',
                        help='Enable per-layer aggregation')
    parser.add_argument('--temporal_avg', action='store_true',
                        help='Enable temporal averaging')
    parser.add_argument('--temporal_window', type=int, default=3,
                        help='Window size for temporal averaging')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Use bootstrapped datasets')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    weight_path = os.path.join(args.output_path, 'weights/')
    os.makedirs(weight_path, exist_ok=True)

    global_model = CLDNN(n_classes=args.n_classes)
    torch.save(global_model.state_dict(), os.path.join(weight_path, 'global_model_weights.pt'))

    dataset_lengths = np.zeros(args.n_devices)
    for i in range(args.n_devices):
        training_data = FederatedDataset(
            device_id=i, n_devices=args.n_devices,
            data_path=args.data_path, mode='train'
        )
        dataset_lengths[i] = len(training_data)

    statedicts = []
    test_losses, test_accs = [], []

    for comm_round in range(args.n_rounds):
        print(f"\n{'='*50}")
        print(f"Communication Round {comm_round + 1}/{args.n_rounds}")
        print(f"{'='*50}")

        state_dict = {}
        accuracies = np.zeros(args.n_devices)
        losses = np.zeros(args.n_devices)

        for dev in range(args.n_devices):
            print(f"\n--- Device {dev + 1}/{args.n_devices} ---")

            training_data = FederatedDataset(
                device_id=dev, n_devices=args.n_devices,
                data_path=args.data_path, mode='train'
            )
            val_data = FederatedDataset(
                device_id=dev, n_devices=args.n_devices,
                data_path=args.data_path, mode='val'
            )
            train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

            device_model = CLDNN(n_classes=args.n_classes)
            sd = torch.load(os.path.join(weight_path, 'global_model_weights.pt'))
            device_model.load_state_dict(sd)

            optimizer = optim.Adam(
                device_model.parameters(),
                lr=args.lr, weight_decay=args.weight_decay
            )
            criterion = nn.CrossEntropyLoss()

            history = train_model(
                train_loader, val_loader, args.epochs, optimizer, criterion,
                device_model, use_temporal_avg=args.temporal_avg,
                device=device, temporal_window=args.temporal_window
            )

            if args.per_layer and comm_round > 0:
                best_wt = per_layer_aggregation(
                    device_model, val_loader, criterion, statedicts, dev,
                    comm_round, device,
                    temporal_avg_fn=temporal_avg if args.temporal_avg else None,
                    use_temporal_avg=args.temporal_avg,
                    temporal_window=args.temporal_window
                )
                print(f"  PL weights: {best_wt}")

            state_dict[dev] = device_model.state_dict()

            test_data = FederatedDataset(
                device_id=dev, n_devices=args.n_devices,
                data_path=args.data_path, mode='test'
            )
            test_loader = DataLoader(test_data, batch_size=1024, shuffle=True)
            testing_loss, testing_accuracy, _, _ = test(
                test_loader, criterion, device_model, device,
                use_temporal_avg=args.temporal_avg,
                temporal_window=args.temporal_window
            )
            accuracies[dev] = testing_accuracy
            losses[dev] = testing_loss

        statedicts.append(state_dict)

        updated_weights = fed_avg(state_dict, args.n_devices, dataset_lengths, device)
        torch.save(updated_weights, os.path.join(weight_path, 'global_model_weights.pt'))

        round_path = os.path.join(weight_path, f'comm_round_{comm_round + 1}/')
        os.makedirs(round_path, exist_ok=True)
        torch.save(updated_weights, os.path.join(round_path, 'global_model_weights.pt'))

        test_losses.append(losses.mean())
        test_accs.append(accuracies.mean())

        print(f"\nRound {comm_round + 1} — Avg Test Acc: {accuracies.mean():.4f}, "
              f"Avg Test Loss: {losses.mean():.4f}")

    print(f"\n{'='*50}")
    print("Training complete.")
    print(f"Final Avg Test Accuracy: {test_accs[-1]:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
