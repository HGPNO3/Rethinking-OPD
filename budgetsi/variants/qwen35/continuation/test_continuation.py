import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
from continuation import validate_continuation
from train import atomic,digest
class ContinuationTest(unittest.TestCase):
 def test_parent_binding_rejects_changed_config_or_artifact(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);parent=p/'parent';parent.mkdir();out=p/'new';out.mkdir();code=p/'code';code.mkdir()
   atomic(parent/'config.json',{'target_nodes':1000,'temperature':.7})
   bound=parent/'bound';bound.write_text('original')
   c={'target_nodes':3000,'temperature':.7,'continuation':{'parent_run':str(parent),'parent_code':str(code),'files_sha256':{str(bound):digest(bound)},'unchanged_source_files':[]}}
   c['temperature']=1
   with self.assertRaisesRegex(AssertionError,'Unexpected scientific'):validate_continuation(c,out)
   c['temperature']=.7;bound.write_text('tampered')
   with self.assertRaisesRegex(AssertionError,'Parent artifact changed'):validate_continuation(c,out)
if __name__=='__main__':unittest.main()
