# TabFIDS v2

Federated Intrusion Detection with Per-Layer Aggregation, Temporal Averaging, and Bootstrapping.

<!-- TODO: Add paper citation once published -->
<!-- If you use this code, please cite: -->

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

This implementation uses the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html).

1. Download the MachineLearningCVE CSV files from the link above.
2. Place them in `raw_data/`:

```
raw_data/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

3. Generate the federated splits:

```bash
# Standard (no bootstrapping)
python dataloader.py --n_devices 11 --frac_train 0.8

# With bootstrapping
python dataloader.py --n_devices 11 --frac_train 0.8 --bootstrap
```

This creates per-device train/val/test pickle files under `data/<n_devices>_nodes/device_<id>/`.

## Usage

### Baseline (FedAvg only, no per-layer, no temporal averaging)

```bash
python main.py --n_devices 11 --n_rounds 10 --epochs 10
```

### With Per-Layer Aggregation

```bash
python main.py --n_devices 11 --n_rounds 10 --epochs 10 --per_layer
```

### With Temporal Averaging

```bash
python main.py --n_devices 11 --n_rounds 10 --epochs 10 --temporal_avg --temporal_window 3
```

### Full (Per-Layer + Temporal Averaging + Bootstrapped Data)

```bash
python main.py --n_devices 11 --n_rounds 10 --epochs 10 --per_layer --temporal_avg --bootstrap
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_devices` | 11 | Number of federated devices |
| `--n_rounds` | 10 | Number of communication rounds |
| `--epochs` | 10 | Local training epochs per round |
| `--n_classes` | 2 | Number of output classes |
| `--batch_size` | 32 | Training batch size |
| `--lr` | 0.0001 | Learning rate |
| `--per_layer` | off | Enable per-layer aggregation |
| `--temporal_avg` | off | Enable temporal averaging |
| `--temporal_window` | 3 | Window size for temporal averaging |
| `--bootstrap` | off | Use bootstrapped datasets |
| `--data_path` | `data/` | Path to processed data |
| `--output_path` | `output/` | Path for weights and logs |
| `--device` | `cuda:0` | Compute device |

## Project Structure

```
tabfidsv2/
├── main.py                 # Main training loop
├── models.py               # CLDNN model architecture
├── aggregation.py          # FedAvg and per-layer aggregation
├── temporal_averaging.py   # Temporal averaging preprocessing
├── train_utils.py          # Training and testing functions
├── dataloader.py           # Dataset classes and creation script
├── requirements.txt
└── README.md
```
