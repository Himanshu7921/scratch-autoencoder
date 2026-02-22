from dataclasses import dataclass

@dataclass
class Config:
    input_dim: int = 28 * 28
    hidden_dim: int = 64
    latent_dim: int = 64
    batch_size: int = 64
    lr: float = 0.001
    epochs: int = 120
    input_path: str = "./data"