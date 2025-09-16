import os 
from PIL import Image
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        self.captions_dict = self.load_captions(captions_file)
        self.image_ids = list(self.captions_dict.keys())

        def load_captions(self, captions_file):
            captions_dict = defaultdict(list)
            with open(captions_file, "r") as f:
                for line in f:
                    img_id, caption = line.strip().split('\t')
                    img_name = img_id.split('#')[0]
                    captions_dict[img_name].append(caption.lower())
            return captions_dict
        
        def __len__(self):
            return len(self.image_ids)
        
        def __getitem__(self, idx):
            img_id = self.image_ids[idx]
            img_path = os.path.join(self.root_dir, img_id)
            image = Image.open(img_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            caption = self.captions_dict[img_id][0]
            numerical_caption = [self.vocab.stoi["<SOS>"]]
            numerical_caption += self.vocab.numericalize(caption)
            numerical_caption.append(self.vocab.stoi["<EOS>"])

            return image, torch.tensor(numerical_caption)
        
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions
