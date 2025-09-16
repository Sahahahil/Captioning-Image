import torch
import torch.nn as nn
import torchvision.models as models

class EncodeCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncodeCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0),-1)
        return self.bn(self.fc(features))
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(inputs)
        return self.fc(outputs)

    def sample(self, features, vocab, max_len=20, beam_width=3):
        """Beam Search Decoding"""
        device = features.device
        sequences = [[list(), 0.0, None]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, state in sequences:
                if len(seq) > 0 and seq[-1] == vocab.stoi["<EOS>"]:
                    all_candidates.append((seq, score, state))
                    continue

                inputs = torch.tensor([seq[-1]] if seq else [vocab.stoi["<SOS>"]]).unsqueeze(0).to(device)
                embeddings = self.emvedb(inputs)
                if state is None:
                    embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
                output, state = self.lstm(embeddings, state)
                preds = self.fc(output.squeeze(1))
                probs = torch.nn.functional.log_softmax(preds, dim=1)

                topk = torch.topk(probs, beam_width, dim=1)
                for i in range(beam_width):
                    candidate = seq + [topk.indices[0][i].item()]
                    candidate_score = score + topk.values[0][i].item()
                    all_candidates.append((candidate, candidate_score, state))

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq = sequences[0][0]
        return " ".join([vocab.itos[idx] for idx in best_seq if idx not in {vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]}])