import logging
import argparse
from pathlib import Path
import torch
import torchaudio
from s3prl.nn import S3PRLUpstream
from s3prl.util.override import parse_overrides
from torch.utils.data import Dataset, DataLoader
import pdb
class AudioDataset(Dataset):
    def __init__(self, scp_file):
        self.data = []
        with open(scp_file, 'r') as f:
            for line in f:
                audio_id, path = line.strip().split()
                self.data.append((audio_id, path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_id, path = self.data[idx]
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate, audio_id

def collate_fn(batch):
    waveforms, sample_rates, audio_ids = zip(*batch)
    lengths = [waveform.shape[1] for waveform in waveforms]
    max_length = max(lengths)
    padded_waveforms = [torch.nn.functional.pad(waveform, (0, max_length - length)) for waveform, length in zip(waveforms, lengths)]
    return torch.stack(padded_waveforms), torch.tensor(lengths), audio_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--scp_file", default="data/audio/wav.scp")
    parser.add_argument("--output_dir", default="data/ssamba_base")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--device", default="cuda")
    args, others = parser.parse_known_args()

    overrides = parse_overrides(others)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    dataset = AudioDataset(args.scp_file)
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False, collate_fn=collate_fn)

    model = S3PRLUpstream(args.name, refresh=args.refresh, extra_conf=overrides).to(
        args.device
    )
    model.eval()

    with torch.no_grad():
        for batch_idx, (waveforms, lengths, audio_ids) in enumerate(dataloader):
            waveforms = waveforms.to(args.device).squeeze(1)
            lengths = lengths.to(args.device)
            hs, hs_len = model(waveforms, lengths)
            hs = [h.detach().cpu() for h, h_len in zip(hs, hs_len)]
            
            # Save each batch's tensor separately
            batch_output_path = output_dir / f"{args.name}_batch_{batch_idx}.pt"
            torch.save(hs, batch_output_path)
            logging.info(f"Saved batch {batch_idx} to {batch_output_path}")

