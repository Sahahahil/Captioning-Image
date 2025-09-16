# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, List


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int, train_backbone: bool = False):
        super(EncoderCNN, self).__init__()
        # Pretrained ResNet50 backbone
        resnet = models.resnet50(pretrained=True)
        # Freeze backbone optionally
        for param in resnet.parameters():
            param.requires_grad_(train_backbone)
        # Remove last fully connected layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return image embeddings of shape (batch, embed_size)"""
        features = self.resnet(images)                    # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)    # (B, 2048)
        features = self.fc(features)                      # (B, embed_size)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Project image features to initial LSTM hidden and cell states
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Training forward"""
        embeddings = self.embed(captions[:, :-1])  # (B, seq_len-1, embed_size)
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        outputs, _ = self.lstm(embeddings, (h0, c0))  # (B, seq_len-1, hidden_size)
        logits = self.fc(outputs)                      # (B, seq_len-1, vocab_size)
        return logits

    def init_hidden_from_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create initial hidden/cell from features"""
        h = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h, c

    def sample(self, features: torch.Tensor, vocab, max_len: int = 20, beam_width: int = 3) -> List[str]:
        """Beam search decoding"""
        device = features.device
        batch_size = features.size(0)
        results = []
        for i in range(batch_size):
            feat = features[i].unsqueeze(0)
            caption = self._beam_search_single(feat, vocab, max_len, beam_width, device)
            results.append(caption)
        return results

    def _beam_search_single(self, feature: torch.Tensor, vocab, max_len: int, beam_width: int, device) -> str:
        """Beam search for a single example"""
        sos, eos, pad = vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi.get("<PAD>")
        h, c = self.init_hidden_from_features(feature)
        h, c = h.to(device), c.to(device)
        sequences = [([sos], 0.0, (h, c))]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, state in sequences:
                if seq[-1] == eos:
                    all_candidates.append((seq, score, state))
                    continue
                last_token = torch.tensor([seq[-1]], dtype=torch.long, device=device).unsqueeze(0)
                emb = self.embed(last_token)
                h_state, c_state = state
                out, (h_new, c_new) = self.lstm(emb, (h_state, c_state))
                logits = self.fc(out.squeeze(1))
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                topk = torch.topk(log_probs, beam_width, dim=1)
                topk_idx, topk_vals = topk.indices[0], topk.values[0]

                for k in range(topk_idx.size(0)):
                    token_idx = int(topk_idx[k].item())
                    token_score = float(topk_vals[k].item())
                    new_seq = seq + [token_idx]
                    new_score = score + token_score
                    new_h, new_c = h_new.detach().clone(), c_new.detach().clone()
                    all_candidates.append((new_seq, new_score, (new_h, new_c)))

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(s[0][-1] == eos for s in sequences):
                break

        best_seq = sequences[0][0]
        words = []
        for idx in best_seq:
            if idx == sos or idx == pad:
                continue
            if idx == eos:
                break
            words.append(vocab.itos.get(idx, "<UNK>"))
        return " ".join(words)

    def greedy_sample(self, features: torch.Tensor, vocab, max_len: int = 20) -> List[str]:
        """Greedy decoding per batch"""
        device = features.device
        results = []
        for i in range(features.size(0)):
            feat = features[i].unsqueeze(0)
            h, c = self.init_hidden_from_features(feat)
            h, c = h.to(device), c.to(device)
            inputs = torch.tensor([[vocab.stoi["<SOS>"]]], dtype=torch.long, device=device)
            caption = []
            for _ in range(max_len):
                emb = self.embed(inputs)
                out, (h, c) = self.lstm(emb, (h, c))
                logits = self.fc(out.squeeze(1))
                pred = logits.argmax(dim=1)
                idx = int(pred.item())
                if idx == vocab.stoi.get("<EOS>"):
                    break
                if idx != vocab.stoi.get("<PAD>") and idx != vocab.stoi.get("<SOS>"):
                    caption.append(vocab.itos.get(idx, "<UNK>"))
                inputs = pred.unsqueeze(1)
            results.append(" ".join(caption))
        return results
