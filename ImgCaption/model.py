import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class CNN(nn.Module):
    def __init__(self, embed_size):
        super(CNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False  
            
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        return features

class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(self.dropout(hiddens))
        return outputs

    def sample(self, features, max_length=20):
        generated_ids = []
        states = None
        inputs = features.unsqueeze(1)
        
        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            generated_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
            if predicted.item() == 1:  
                break
                
        return generated_ids

class Caption(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Caption, self).__init__()
        self.cnn = CNN(embed_size)
        self.rnn = RNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.cnn(images)
        outputs = self.rnn(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_length=20):
        features = self.cnn(image)
        caption_ids = self.rnn.sample(features, max_length=max_length)
        caption = [vocab.idx2word[idx] for idx in caption_ids]
        return ' '.join(caption)

