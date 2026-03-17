import os
import datetime
import numpy as np
import torch
from torch import optim
from training.loss import cox_loss
from utils.logging import get_logger
from lifelines.utils import concordance_index


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, logger):
    """
    Performs one epoch of model training on the standalone BINN, logging cox loss
    
    Args:
        model (torch.nn.Module): model object
        loader (torch DataLoader): the dataloader to train on
        optimizer (torch Optimizer): the optimizer class to use
        loss_fn (function): the custom cox loss function
        device (torch.cuda.device): train on gpu or cpu
        logger (Logger): logger object
    """
    
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    
    for batch_num, batch in enumerate(dataloader):   # dataloader __getitem__ returns dict
        X = batch['X_mapped'].to(device)    # TEMP for standalone BINN, we only use mapped genes
        y_time = batch['y_time'].to(device)
        y_event = batch['y_event'].to(device)
        batch_size = len(y_event)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y_time, y_event)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch_num % (num_batches//3) == 0:
            loss = loss.item()
            train_loss += loss
            current = batch_num * batch_size + batch_size
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    logger.info(f"Average Training Loss: {train_loss:>8f}")
    return train_loss


def evaluate(model, dataloader, loss_fn, device, logger):
    """
    Performs one epoch of model validation on the standalone BINN, logging validation loss and overall C-index
    
    Args:
        model (torch.nn.Module): model object
        loader (torch DataLoader): the dataloader to evaluate on (e.g. val or test)
        loss_fn (function): the custom cox loss function
        device (torch.cuda.device): train on gpu or cpu
        logger (Logger): logger object
    
    Returns:
        val_loss (float): average loss across the set
        cindex (float): overall C-Index across the set
    """
    
    model.eval()
    val_loss = 0
    all_preds, all_times, all_events = [], [], []

    # run validation set (no need for gradients)
    with torch.no_grad():
        for batch in dataloader:
            X = batch['X_mapped'].to(device)    # TEMP for standalone BINN, we only use mapped genes
            y_time = batch['y_time'].to(device)
            y_event = batch['y_event'].to(device)
            
            pred = model(X)
            val_loss += loss_fn(pred, y_time, y_event).item()
            
            all_preds.append(pred.cpu().numpy())
            all_times.append(y_time.cpu().numpy())
            all_events.append(y_event.cpu().numpy())

    val_loss /= len(dataloader)
    
    preds  = np.concatenate(all_preds)
    times  = np.concatenate(all_times)
    events = np.concatenate(all_events)
    
    cindex = concordance_index(times, -preds, events)  # negative preds since higher risk = lower survival time

    return val_loss, cindex


def train(model, train_loader, val_loader, train_proportion, val_proportion, 
          logfile, epochs=10, alpha=1e-3, weight_decay=1e-4, stop_early_patience=5):
    """
    Outer training loop that initiates log and runs epochs of train and val, stopping early if needed
    
    Args:
        model (torch.nn.Module): model object
        train_loader (torch DataLoader): dataloader for the training set
        val_loader (torch DataLoader): dataloader for the validation set
        train_proportion (float): train split proportion for logging purposes
        val_proportion (float): val split proportion for logging purposes
        logfile (str): path to the .log file (in experiments/runs/...) to log results to
        # TODO: read config file
        **epochs (int): number of epochs for training (default=10)
        **alpha (float): learning rate for Adam optimizer (default=1e-3)
        **weight_decay (float): weight_decay for Adam optimizer (default=1e-4)
    
    Returns:
        train_losses (list): average training loss of each epoch
        val_losses (list): average validation loss of each epoch
        cindexes (list): validation C-Index of each epoch
    """
    
    # Start log
    log_path = os.path.abspath(logfile)
    logger = get_logger("trainer", log_path)
    logger.info("=========================================")
    logger.info(f"trainer.py log started {datetime.datetime.now()}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"\tin_nodes: {model.sc1.weight.shape[1]}")
    logger.info(f"\tpathway_nodes: {model.sc2.weight.shape[1]}")
    logger.info(f"\thidden_nodes: {model.sc3.weight.shape[1]}")
    logger.info(f"\tout_nodes: {model.sc4.weight.shape[1]}")
    # in_nodes, pathway_nodes, hidden_nodes, out_nodes
    logger.info(f"Train: {train_proportion} = {len(train_loader.dataset)} samples")
    logger.info(f"Validation: {val_proportion} = {len(val_loader.dataset)} samples")
    logger.info(f"Batch Size: {train_loader.batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Alpha: {alpha}")
    logger.info(f"Weight_Decay: {weight_decay}")
    
    # Prepare to train
    optimizer = optim.Adam(model.parameters(), 
                           lr=alpha, weight_decay=weight_decay)  # weight decay applies L2 reg
    loss_fn = cox_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Stop early params
    best_cindex = 0
    best_epoch = -1
    patience_counter = 0
    patience = stop_early_patience
    
    # Track epoch losses to return for plotting
    train_losses = []
    val_losses = []
    cindexes = []

    for epoch in range(epochs):
        
        # train
        logger.info(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, logger)
        val_loss, cindex = evaluate(model, val_loader, loss_fn, device, logger)
        logger.info(f"Val Loss: {val_loss:.4f} | C-index: {cindex:.4f}\n")  # log evaluation outside of function for now
        
        # store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        cindexes.append(cindex)
        
        # check if we should stop early
        if cindex > best_cindex:
            best_cindex = cindex
            best_epoch = epoch+1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(os.path.dirname(logfile), "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Stopping early at Epoch {epoch+1} since no C-Index improvement after {patience} epochs.")
                break
            
    logger.info("Done!")
    logger.info(f"Best C-Index = {best_cindex} at Epoch {best_epoch}")
    
    return train_losses, val_losses, cindexes


def test(model, test_loader, best_model_file, logfile):
    """
    Separate testing loop to be run after training, getting a final unbiased evaluation of the model
    
    Args:
        model (torch.nn.Module): model object
        test_loader (torch DataLoader): dataloader for the testing set
        best_model_file (str): path to the best_model.pt file to load and test
        logfile (str): path to the .log file (in experiments/runs/...) to log results to
    
    Returns:
        test_losses (list): average testing loss
        test_cindex (float): overall C-Index of the test set
    """
    
    loss_fn = cox_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = os.path.abspath(logfile)
    logger = get_logger("tester", log_path)
    
    # Load and test model
    model.load_state_dict(torch.load(best_model_file))
    avg_test_loss, test_cindex = evaluate(model, test_loader, loss_fn, device, logger)
    
    logger.info(f"FINAL TEST Average Loss: {avg_test_loss:.4f}")
    logger.info(f"FINAL TEST C-index: {test_cindex:.4f}")
    
    return avg_test_loss, test_cindex