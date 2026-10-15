import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x shape: (N, C, H, W) -> permute to (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # permute back to (N, C, H, W)
        return x.permute(0, 3, 1, 2)

class Block(nn.Module):
    def __init__(self, num_channels, kernel_size):
        """similar to ResNet block but with LayerNorm instead of BatchNorm"""
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=1),
            LayerNorm2d(num_channels),
            nn.Tanh(),
            nn.Conv2d(num_channels, num_channels, kernel_size, padding=1),
            LayerNorm2d(num_channels),
        )
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.seq(x) + x)
        
class Model(nn.Module):
    def __init__(self, num_channels, kernel_size, num_blocks, rows, cols):
        super().__init__()

        self.C = num_channels
        self.K = kernel_size
        self.B = num_blocks
        self.rows = rows
        self.cols = cols 

        # ResNet blocks to capture features
        self.features = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size, padding=1) # kernel_size=3 and padding=1 retains height and width
        )
        for _ in range(num_blocks):
            self.features.append(Block(num_channels, kernel_size))

        # convolution to collapse channels followed by hidden layer to learn non-linearity in assessing scalar value of position
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1, padding=0), # reduce channels to 1
            nn.ReLU(),
            nn.Flatten(), # merges every dim except B so gives 2D tensor
            nn.Linear(rows * cols, 256), # expand
            nn.ReLU(),
            nn.Linear(256, 1), # contract
            nn.Tanh()
        )

        # convolution to collapse channels followed by linear layer to map to actions vector
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, 1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * rows * cols, cols)
        )
        
    def forward(self, x) -> (torch.tensor, torch.tensor):
        x = self.features(x)  # (B, C, H, W)

        # get value scalar [-1, 1]
        val = self.value_head(x) # [B, 1]
        
        # get action logits
        action_logits = self.policy_head(x) # [B, 7]
        
        return val, action_logits

    def params_count(self):
        return sum(x.numel() for x in self.parameters())

    def save(self, path):
        torch.save({
                "config": {"num_channels": self.C, "kernel_size": self.K, "num_blocks": self.B, "rows": self.rows, "cols": self.cols},
                "state_dict": self.state_dict()
            }, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, weights_only=True) # plain dicts/ints are allowed with weights_only=True
        model = cls(**checkpoint["config"]) # call constructor
        model.load_state_dict(checkpoint["state_dict"]) # load weights
        return model # return newly instantiated object


def test_model_overfits(model, optimizer):
    # dummy datapoint of a mid-point game (from a real self_play example
    x = torch.tensor([[[[ 0.,  0.,  1.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  1.,  0.,  0.,  0.,  0.],
              [ 0.,  0., -1., -1.,  0., -1., -1.],
              [ 0.,  0.,  1.,  1.,  0., -1.,  1.],
              [ 0.,  0.,  1., -1.,  0.,  1., -1.],
              [ 1.,  1.,  1., -1., -1.,  1., -1.]]]])
    val = torch.tensor([[-0.6958]])
    probs = torch.tensor([[0.0910, 0.0928, 0.01, 0.161, 0.4599, 0.0957, 0.0896]])
    
    # optimization loop
    for i in range(1000):
        pred_val, pred_probs = model(x)
        
        optimizer.zero_grad()
        val_loss, act_loss = compute_loss(v=pred_val, p=pred_probs, z=val, pi=probs)
        loss = val_loss + act_loss
        loss.backward()
        optimizer.step()
    
    print(val, pred_val)
    val_is_equal = torch.allclose(val, pred_val, rtol=1e-5, atol=1e-8)
    
    print(probs, F.softmax(pred_probs, dim=1))
    probs_are_equal = torch.allclose(probs, F.softmax(pred_probs, dim=1), rtol=1e-5, atol=1e-8)

    return val_is_equal, probs_are_equal, loss  # 1.6571 is minimum loss

def compute_loss(v, p, z, pi):
    """
    Loss = MSE(root value, policy value) + Cross-Entropy(root's children normalized visit counts, policy action probs) + ||weights||^2
    Note: using AdamW we already get weight decay/regularization so don't need to add regularizer term
    Note: prediction must be first term in F.cross_entropy

    v, p come from policy neural net
    z, pi come from MCTS synthetic games
    
    works with batch dimension:
    v, z -> [B, 1]
    p, pi -> [B, 7]
    """
    value_loss = (v - z).pow(2).mean()
    policy_loss = F.cross_entropy(p, pi)
    return value_loss, policy_loss