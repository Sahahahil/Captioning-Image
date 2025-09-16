import os
import csv
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None, mode="train"):
        """
        root_dir: directory containing images
        captions_file: path to captions file (.csv or .tsv). If CSV, expected columns: image,caption
        vocab: Vocabulary instance (must contain <SOS>, <EOS>, <PAD>, <UNK>)
        transform: torchvision transforms applied to PIL image
        mode: "train" -> pick random caption per image; "val"/"test" -> use first caption deterministically
        """
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        self.mode = mode

        # quick checks
        required_tokens = ["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
        for t in required_tokens:
            if t not in getattr(self.vocab, "stoi", {}):
                raise ValueError(f"Vocabulary missing required token: {t}")

        # build set/list of image files available (lowercased)
        self.image_files = {}
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        for fname in os.listdir(self.root_dir):
            _, ext = os.path.splitext(fname)
            if ext.lower() in image_extensions:
                self.image_files[fname.lower()] = fname  # map lower -> actual filename

        # load captions (only keep entries for which an image exists)
        self.captions_dict, self.skipped = self.load_captions(captions_file)

        self.image_ids = list(self.captions_dict.keys())

        # Debug info
        print(f"📊 Dataset loaded: {len(self.image_ids)} unique images")
        if len(self.image_ids) > 0:
            sample_img = self.image_ids[0]
            print(f"   Sample image: {sample_img}")
            print(f"   Captions for sample: {len(self.captions_dict[sample_img])}")
        if self.skipped:
            print(f"⚠️  Skipped {len(self.skipped)} caption rows due to missing images or parse errors (see first 5):")
            for s in self.skipped[:5]:
                print("   ", s)

    def load_captions(self, captions_file):
        """
        Returns (captions_dict, skipped_rows)
        captions_dict: {image_filename_actual_case: [caption1, caption2, ...]}
        skipped_rows: list of raw rows/lines skipped
        """
        captions_dict = defaultdict(list)
        skipped = []

        ext = os.path.splitext(captions_file)[1].lower()

        def process_row(img_id_raw, caption_raw):
            # remove trailing #0/#1 etc (common in Flickr files)
            img_id = img_id_raw.split("#")[0].strip()
            caption = caption_raw.strip().lower()

            # normalize filename to lower for matching; try as-is and with common extensions
            candidates = [img_id, img_id.lower()]
            if not os.path.splitext(img_id)[1]:
                # if no ext in caption file, try .jpg and .jpeg
                candidates += [img_id + ".jpg", img_id + ".jpeg", (img_id + ".JPG").lower()]
            # add common case variants
            candidates = [c.lower() for c in candidates]

            found = False
            for c in candidates:
                if c in self.image_files:
                    actual_fname = self.image_files[c]
                    captions_dict[actual_fname].append(caption)
                    found = True
                    break

            if not found:
                skipped.append((img_id_raw, caption_raw))

        try:
            if ext == ".csv":
                with open(captions_file, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    # peek the first row to determine header
                    try:
                        first = next(reader)
                    except StopIteration:
                        return dict(), []
                    # if header contains 'image' and 'caption', skip it
                    header = [c.strip().lower() for c in first]
                    if len(header) >= 2 and header[0] == "image":
                        # header present -> proceed with remaining rows
                        pass
                    else:
                        # first row is actually data -> process it
                        if len(first) >= 2:
                            process_row(first[0], first[1])
                    for row in reader:
                        if len(row) >= 2:
                            process_row(row[0], row[1])
                        else:
                            skipped.append(tuple(row))
            else:
                # treat as TSV or generic tab-separated caption file
                with open(captions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # support optional CSV header line starting with 'image,caption'
                        if line.lower().startswith("image,caption"):
                            # skip header
                            continue
                        # prefer splitting on tab first, else on first comma
                        if "\t" in line:
                            parts = line.split("\t", 1)
                        elif "," in line:
                            parts = line.split(",", 1)
                        else:
                            parts = []
                        if len(parts) == 2:
                            process_row(parts[0], parts[1])
                        else:
                            skipped.append((line,))
        except Exception as e:
            raise RuntimeError(f"Error reading captions file '{captions_file}': {e}")

        # convert defaultdict to plain dict
        captions_dict = dict(captions_dict)
        return captions_dict, skipped

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Returns: image_tensor, caption_tensor (LongTensor)
        caption_tensor contains: [<SOS>, ...tokens..., <EOS>]
        """
        img_actual_fname = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, img_actual_fname)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # log and return a fallback image (but don't hide errors silently)
            print(f"❌ Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224, 224), color="black")

        # choose caption
        captions = self.captions_dict[img_actual_fname]
        if len(captions) == 0:
            caption = ""
        else:
            if self.mode == "train":
                # random caption each call -> variability for training
                idx_choice = torch.randint(0, len(captions), (1,)).item()
                caption = captions[idx_choice]
            else:
                # deterministic for val/test
                caption = captions[0]

        if self.transform:
            image = self.transform(image)

        # numericalize: add SOS and EOS
        sos = self.vocab.stoi["<SOS>"]
        eos = self.vocab.stoi["<EOS>"]
        num_caption = [sos] + self.vocab.numericalize(caption) + [eos]
        caption_tensor = torch.tensor(num_caption, dtype=torch.long)

        return image, caption_tensor


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        """
        batch: list of tuples (image_tensor, caption_tensor)
        returns images (B, C, H, W), captions (B, max_len)
        """
        images = torch.stack([item[0] for item in batch], dim=0)
        captions = [item[1] for item in batch]
        captions_padded = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.pad_idx
        )
        return images, captions_padded
