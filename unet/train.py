from utility.seed import set_seed

set_seed(77)

import mlflow
import time
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from custom_data.drive_data import DriveDataset
from model.model import build_unet
from utility.loss import DiceBCELoss, DiceLoss
from utility.utils import make_dir, epoch_time

def train(model, loader, optimizer, loss_function, device):
    """
    """
    epoch_loss = 0.0

    model.train()

    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)

    return epoch_loss

def evaluate(model, loader, loss_function, device):
    """
    """

    epoch_loss = 0.0

    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_function(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)

    return epoch_loss

if __name__ == '__main__':

    set_seed(77)

    # Hyperparameters - Create config file separately
    HEIGHT = 512
    WEIGHT = 512
    SIZE = (HEIGHT, WEIGHT)
    BATCH_SIZE = 1
    EPOCHS = 35
    LEARNING_RATE = 1e-4

    # Paths to save weights
    make_dir(r'weights\\')
    checkpoint_path = r"weights\checkpoint.pth"

    # Train Validate Data Split
    train_x = glob(r'..\dataset\augmented\training\images\*')
    train_y = glob(r'..\dataset\augmented\training\annotated\*')
    valid_x = glob(r'..\dataset\augmented\validate\images\*')
    valid_y = glob(r'..\dataset\augmented\validate\annotated\*')

    # Load Dataset
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    # Train Loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    # Validate Loader
    validate_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # Initialize model
    device = torch.device('cuda')

    # Release GPU memory
    torch.cuda.empty_cache()

    print(f'total mem: {torch.cuda.get_device_properties(0).total_memory}')
    print(f'mem reserved: {torch.cuda.max_memory_reserved(0)}')
    print(f'mem allocated: {torch.cuda.memory_allocated(0)}')

    model = build_unet()
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            'min',
                                                            patience=5,
                                                            verbose=True)

    loss_fn = DiceBCELoss()

    best_valid_loss = float("inf")

    # Using mlflow
    experiment_name = "unet"

    # Define params used
    params = {'learning_rate': LEARNING_RATE, 
            'batch size': BATCH_SIZE,
            'epoch': EPOCHS}
    try:
        # creating a new experiment
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=exp_id, run_name='second_run'):

        mlflow.log_params(params)

        # Training the model
        for epoch in tqdm(range(EPOCHS)):
            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, loss_fn, device)
            valid_loss = evaluate(model, validate_loader, loss_fn, device)

            # Saving the model
            if valid_loss < best_valid_loss:
                print(f"best valid loss so far: {valid_loss:2.4f}")

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saving checkpoint for best valid loss: {best_valid_loss}")

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s'
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\tValidate Loss: {valid_loss:.3f}\n'

            # Logging the metrics
            metrics ={'epoch': epoch+1,
                    'epoch min': epoch_mins,
                    'epoch sec': epoch_secs,
                    'train loss': train_loss,
                    'validate loss': valid_loss,
                    'best valid loss': best_valid_loss}

            mlflow.log_metrics(metrics=metrics)

            # print(f'print summary mem: {torch.cuda.memory_summary()}')

            print(data_str)

# Epoch 32 training loss 0.902, best validate loss 1.049