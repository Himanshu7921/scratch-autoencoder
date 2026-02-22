import torch
import torch.nn.functional as F
from tqdm import tqdm
from os.path  import join

from .model import AutoEncoder
from .config import Config
from .data_loader import MnistDataloader


config = Config()
training_images_filepath = join(config.input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(config.input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(config.input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(config.input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


def train_auto_encoder(auto_encoder, x, optimizer, epochs, batch_size=32):
    auto_encoder.train()

    # Dataloader
    dataset = torch.utils.data.TensorDataset(x)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in epoch_bar:
            batch = batch[0]
            output = auto_encoder(batch)
            loss = F.mse_loss(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # live batch-wise loss inside progress bar
            epoch_bar.set_postfix(loss=loss.item())

        # print loss at interval
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

if __name__ == "__main__":
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    x_train = x_train / 255.0
    x_train = x_train.view(x_train.shape[0], -1)

    # Get the Model Ready
    auto_encoder = AutoEncoder(
                    input_message_dim = config.input_dim,
                    enc_hidden_dim = config.hidden_dim,
                    encoded_message_dim = config.latent_dim,
                    dec_hidden_dim = config.hidden_dim
    )
    optimizer = torch.optim.Adam(params = auto_encoder.parameters(), lr = config.lr)

    train_auto_encoder(
        auto_encoder = auto_encoder,
        x = x_train,
        optimizer = optimizer,
        epochs = config.epochs,
        batch_size = config.batch_size
    )

    # Save the Model After Training
    torch.save(auto_encoder.state_dict(), "autoencoder_weights.pth")
    print("\nTraining completed. The model has been saved to 'autoencoder_weights.pth'.")