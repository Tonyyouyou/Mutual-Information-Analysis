from espnet2.bin.asr_inference import Speech2Text
import torch
import os
import soundfile
import pickle
import pdb
class Speech2TextWithHooks(Speech2Text):
    def __init__(self, *args, save_dir="./layer_outputs", **kwargs):
        super().__init__(*args, **kwargs)
        self.representations = {}
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.register_hooks()

    def register_hooks(self):
        def get_representation_hook(layer_name):
            def hook(module, input, output):
                # pdb.set_trace()
                if isinstance(output, tuple):
                    output = output[0]
                self.representations[layer_name] = output.detach()
            return hook
        
        # # Hook for feature_extractor output
        # self.asr_model.frontend.upstream.upstream.model.feature_extractor.register_forward_hook(
        #     get_representation_hook("feature_extractor"))

        # Hooks for each ConBiMambaWav2Vec2EncoderLayer
        for i, layer in enumerate(self.asr_model.frontend.upstream.upstream.model.encoder.layers):
            layer.register_forward_hook(
                get_representation_hook(f"ConBiMambaWav2Vec2EncoderLayer_{i}")
            )

        self.asr_model.frontend.upstream.upstream.model.encoder.register_forward_hook(
            get_representation_hook("TransformerEncoder"))
        
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        results = super().__call__(*args, **kwargs)

        # Save the collected representations to a pickle file
        save_path = os.path.join(self.save_dir, "layer_outputs.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(self.representations, f)
        
        # Optionally print the shapes of the saved representations
        for layer_name, representation in self.representations.items():
            print(f"Layer: {layer_name}, Representation Shape: {representation.shape}")
        
        return results

asr_train_config="/home/xyz/Desktop/espnet/egs2/librispeech_100/asr1/exp/asr_train_mhubert_conformer_raw_en_bpe5000/config.yaml"
asr_model_file="/home/xyz/Desktop/espnet/egs2/librispeech_100/asr1/exp/asr_train_mhubert_conformer_raw_en_bpe5000/valid.acc.ave_10best.pth"

model = Speech2TextWithHooks(asr_train_config,asr_model_file, device='cuda')



audio = "/home/xyz/Downloads/Data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

speech, rate = soundfile.read(audio)

results = model(speech)
