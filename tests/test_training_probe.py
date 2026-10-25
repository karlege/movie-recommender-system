"""Check target isolation, exact prefix membership, and diagnostic neutrality."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.data import Interaction, UserSplit
from src.dataset import MovieVocabulary, PrefixDataset, build_evaluation_examples, main as prepare
from src.training_probe import build_training_probe
from src.train import TrainConfig, run_training


class TrainingProbeTests(unittest.TestCase):
    def test_probe_is_trained_prefix_and_masks_only_earlier_raw_ratings(self):
        s = UserSplit(1, tuple(Interaction(i,t) for i,t in [(10,10),(20,20),(30,30)]),
                      Interaction(40,40), Interaction(50,50))
        splits = {1:s}
        # Candidate catalogue may contain items learned from other users.
        vocab = MovieVocabulary([10,20,25,30,35,40,50])
        ratings = pd.DataFrame([(1,10,5,10),(1,20,5,20),(1,25,1,30),
                                (1,30,5,30),(1,35,1,30),(1,40,5,40),(1,50,5,50)],
                               columns=['userId','movieId','rating','timestamp'])
        val,_ = build_evaluation_examples(ratings,splits,vocab,'validation',1)
        records,meta = build_training_probe(ratings,splits,vocab,val,1)
        probe = records[0]
        sample = PrefixDataset(splits,vocab,1)[-1]
        np.testing.assert_array_equal(probe.history_ids,sample['history_ids'])
        self.assertEqual(probe.target_id,sample['target_ids'])
        self.assertEqual(probe.length,sample['lengths'])
        self.assertEqual([vocab.decode(int(i)) for i in probe.seen_ids],[10,20,25])
        self.assertEqual(splits[1],s)
        self.assertEqual(meta['users'],1)
        self.assertEqual(vocab.decode(probe.target_id),30)
        self.assertEqual(vocab.decode(val[0].target_id),40)

    def test_probe_rejects_trial_training(self):
        with self.assertRaisesRegex(ValueError,'full training'):
            TrainConfig(log_train_recall=True,max_train_examples=4).validate()

    def test_logging_does_not_change_training_or_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            rows=[(1,10,5,10),(1,20,5,20),(1,30,5,30),(1,40,5,40),(1,50,5,50),
                  (2,40,5,10),(2,50,5,20),(2,60,5,30),(2,10,5,40),(2,20,5,50)]
            (raw/'ratings.dat').write_text('\n'.join('::'.join(map(str,r)) for r in rows))
            (raw/'movies.dat').write_text('\n'.join(f'{i}::Movie {i}::Drama' for i in range(10,61,10)))
            with patch.object(sys,'argv',['prepare','--data-dir',str(raw),'--output-dir',str(root/'processed')]),contextlib.redirect_stdout(io.StringIO()):
                prepare()
            for model in ('mlp','transformer'):
                reports=[];histories=[];states=[]
                for enabled in (False,True):
                    path=root/f'{model}_{enabled}'
                    cfg=TrainConfig(model=model,epochs=2,batch_size=3,eval_batch_size=2,
                                    num_threads=1,d_model=8,n_layers=1,mlp_hidden=12,
                                    dropout=.1,log_train_recall=enabled)
                    reports.append(run_training(raw,root/'processed/preparation.json',path,cfg,False))
                    histories.append([json.loads(x) for x in (path/'history.jsonl').read_text().splitlines()])
                    states.append(torch.load(path/'best.pt',weights_only=True)['model_state_dict'])
                self.assertEqual(reports[0]['best_epoch'],reports[1]['best_epoch'])
                for a,b in zip(*histories):
                    self.assertEqual(a['train_loss'],b['train_loss'])
                    self.assertEqual(a['metrics'],b['metrics'])
                    self.assertEqual(b['train_metrics']['users'],2)
                for key in states[0]:
                    torch.testing.assert_close(states[0][key],states[1][key],rtol=0,atol=0)
                (root/f'{model}_True').rename(root/f'{model}_20ep_seed42_l4')
            from src.plot_train_recall import main as plot
            with patch.object(sys,'argv',['plot','--runs-root',str(root),
                                         '--output-dir',str(root/'figures'),'--seeds','42']),contextlib.redirect_stdout(io.StringIO()):
                plot()
            self.assertTrue((root/'figures/all_train_validation_recall.pdf').is_file())



if __name__=='__main__':
    unittest.main()
