import json,tempfile,unittest
from pathlib import Path
from budgetsi.model_runtime import eos_contract
class Tokenizer:
    vocab={'<|im_start|>':0,'<|im_end|>':1,'<|endoftext|>':2,'<think>':3,'</think>':4}
    def get_vocab(self):return self.vocab
    def encode(self,text,**kwargs):return [self.vocab[text]]
class MetadataTests(unittest.TestCase):
    def test_missing_generation_config_uses_text_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'config.json').write_text(json.dumps({'text_config':{'vocab_size':5,'eos_token_id':1}}))
            self.assertEqual(eos_contract(Tokenizer(),p),{1})
            (p/'generation_config.json').write_text(json.dumps({'eos_token_id':[1,2]}))
            self.assertEqual(eos_contract(Tokenizer(),p),{1,2})
if __name__=='__main__':unittest.main()
