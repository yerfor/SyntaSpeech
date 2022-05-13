import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.tts.syntaspeech.syntaspeech import SyntaSpeech
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams

from modules.tts.syntaspeech.syntactic_graph_buider import Sentence2GraphParser

class SyntaSpeechInfer(BaseTTSInfer):
    def __init__(self, hparams, device=None):
        super().__init__(hparams, device)
        if hparams['ds_name'] in ['biaobei']:
            self.syntactic_graph_builder = Sentence2GraphParser(language='zh')
        elif hparams['ds_name'] in ['ljspeech', 'libritts']:
            self.syntactic_graph_builder = Sentence2GraphParser(language='en')

    def build_model(self):
        ph_dict_size = len(self.ph_encoder)
        word_dict_size = len(self.word_encoder)
        model = SyntaSpeech(ph_dict_size, word_dict_size, self.hparams)
        load_ckpt(model, hparams['work_dir'], 'model')
        model.to(self.device)
        with torch.no_grad():
            model.store_inverse_all()
        model.eval()
        return model

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        word_tokens = torch.LongTensor(item['word_token'])[None, :].to(self.device)
        word_lengths = torch.LongTensor([word_tokens.shape[1]]).to(self.device)
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)
        dgl_graph, etypes = self.syntactic_graph_builder.parse(item['text'], words=item['words'].split(" "), ph_words=item['ph_words'].split(" "))
        dgl_graph = dgl_graph.to(self.device)
        etypes = etypes.to(self.device)
        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'word_tokens': word_tokens,
            'word_lengths': word_lengths,
            'ph2word': ph2word,
            'spk_ids': spk_ids,
            'graph_lst': [dgl_graph],
            'etypes_lst': [etypes]
        }
        return batch
    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        with torch.no_grad():
            output = self.model(
                sample['txt_tokens'],
                sample['word_tokens'],
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                infer=True,
                forward_post_glow=True,
                spk_id=sample.get('spk_ids'),
                graph_lst=sample['graph_lst'],
                etypes_lst=sample['etypes_lst']
            )
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]


if __name__ == '__main__':
    SyntaSpeechInfer.example_run()
