import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from model import Caption  

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab

        with open(captions_file, 'r') as file:
            lines = file.readlines()
        self.image_captions = []
        for line in lines:
            img, caption = line.strip().split('\t')
            self.image_captions.append((img, caption))

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        img_name, caption = self.image_captions[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tokens = [self.vocab.word2idx["<start>"]] + \
                 [self.vocab.word2idx.get(word, self.vocab.word2idx["<unk>"]) for word in caption.split()] + \
                 [self.vocab.word2idx["<end>"]]
        return image, torch.tensor(tokens)

# Vocabulary class
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocab(self, sentences):
        for sentence in sentences:
            for word in sentence.split():
                self.add_word(word)

# Define transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Training function
def train(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, captions = images.to(device), captions.to(device)

            outputs = model(images, captions)
            print(outputs.size)

            targets = captions[:, 1:]  # Ignore <start> token

            lengths = (targets != vocab.word2idx["<pad>"]).sum(dim=1)
            packed_targets = torch.nn.utils.rnn.pack_padded_sequence(
                targets, lengths.cpu(), batch_first=True, enforce_sorted=False
            ).data

            outputs = outputs.view(-1, outputs.size(2))  # Flatten outputs to (batch_size * seq_len, vocab_size)

            print(outputs.size)
            packed_targets = packed_targets.view(-1)

            loss = criterion(outputs, packed_targets)

            optimizer.adam()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}")

# Main script
if __name__ == "__main__":
    image_dir = "./Flickr8k_Dataset/Flicker8k_Dataset/"  # Path to image directory
    captions_file = './flickr8k_captions_cleaned.txt' 
    
    with open(captions_file, 'r') as file:
        lines = [line.strip().split('\t')[1] for line in file.readlines()]
    vocab = Vocabulary()
    vocab.add_word("")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")
    vocab.add_word("<pad>")
    vocab.build_vocab(lines)

    dataset = Flickr8kDataset(image_dir, captions_file, vocab, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: (
        torch.stack([i[0] for i in x]),  
        nn.utils.rnn.pad_sequence([i[1] for i in x], batch_first=True, padding_value=vocab.word2idx["<pad>"])  
    ))

    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab.word2idx)
    num_layers = 1
    num_epochs = 10
    model = Caption(embed_size, hidden_size, vocab_size, num_layers)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    train(model, dataloader, criterion, optimizer, device, num_epochs)

    torch.save(model.state_dict(), "image_captioning_model.pth")

