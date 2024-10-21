# models.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import logging

def train(model: nn.Module, dataloader: DataLoader, epochs: int = 10, learning_rate: float = 0.001):
    """Train the formula generator model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            knowns, target_var, target_value = batch
            input_tensor = torch.tensor([[knowns[var] for var in sorted(knowns)] for knowns in knowns]) 
            target_tensor = torch.tensor(target_value).unsqueeze(1)

            optimizer.zero_grad()
            output = model(input_tensor.float()) 
            loss = criterion(output, target_tensor.float())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        logging.info(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    logging.info('Training complete.')
