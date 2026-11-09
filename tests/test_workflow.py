"""Offline tests; all small numeric inputs are synthetic and created temporarily."""

import ast
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from workflow_config import input_errors, load_settings, model_filename, safe_error, selected_cities, WorkflowSettings


class FakeAgent:
    """Minimal conversational stand-in that cannot call a provider."""

    def __init__(self, name="test", responses=None, **kwargs):
        self.name = name
        self.responses = list(responses or ["{}"])
        self.kwargs = kwargs
        self.chat_messages = {}
        self.sent = []

    def initiate_chat(self, recipient, message, **kwargs):
        if not isinstance(message, str):
            raise AssertionError("The transport message must be a string.")
        self.sent.append(message)
        response = recipient.responses.pop(0) if len(recipient.responses) > 1 else recipient.responses[0]
        self.chat_messages[recipient] = [{"content": response}]
        return SimpleNamespace(chat_history=[{"content": response}])


def definitions():
    notebook = json.loads((ROOT / "MulitiCityStudy.ipynb").read_text(encoding="utf-8"))
    namespace = {"pd": pd, "np": np, "json": json, "re": re, "io": io, "os": os,
                 "Dict": dict, "Any": object, "List": list, "Optional": __import__('typing').Optional,
                 "ConversableAgent": FakeAgent, "UserProxyAgent": FakeAgent,
                 "GradientBoostingRegressor": GradientBoostingRegressor, "MDS": MDS,
                 "StandardScaler": StandardScaler, "r2_score": r2_score,
                 "safe_error": safe_error,
                 "read_from_csv": lambda path: pd.read_csv(path).to_csv(index=False)}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(''.join(cell["source"]))
        only_definitions = ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))],type_ignores=[])
        exec(compile(only_definitions,"offline_notebook_definitions","exec"), namespace)
    return namespace


@contextmanager
def private_directory():
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        try:
            yield Path(folder)
        finally:
            os.chdir(original)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.ns = definitions()

    def test_notebook_is_code_only_and_has_no_trial_matrix(self):
        nb = json.loads((ROOT / 'MulitiCityStudy.ipynb').read_text(encoding='utf-8'))
        code = []
        for cell in nb['cells']:
            self.assertNotIn('attachments',cell)
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[])
                self.assertIsNone(cell['execution_count'])
                tree=ast.parse(''.join(cell['source']))
                self.assertFalse(any(isinstance(n,ast.Constant) and n.value is Ellipsis for n in ast.walk(tree)))
                code.append(''.join(cell['source']))
        self.assertEqual(len(code),8)
        self.assertNotIn('workflow.baselines', '\n'.join(code))
        self.assertNotIn('workflow.llm_experiments', '\n'.join(code))

    def test_population_conservation_and_unique_ids(self):
        with private_directory():
            Path('ACS').mkdir()
            pd.DataFrame({'Estimate!!Total!!Total population':[100]}).to_csv('ACS/00_001.csv',index=False)
            cohort={'cluster_id':'c1','population_count':100,'demographic_summary':'synthetic', 'behavior_archetype':'synthetic'}
            candidate={'stage':'population','clusters':[cohort]}
            check=self.ns['hard_check_population']
            self.assertTrue(check(candidate,'00','001')['ok'])
            cohort['population_count']=102
            self.assertTrue(check(candidate,'00','001')['ok'])
            cohort['population_count']=103
            self.assertFalse(check(candidate,'00','001')['ok'])
            candidate['clusters'].append(dict(cohort))
            self.assertIn('cluster_id_not_unique',check(candidate,'00','001')['violations'])

    def test_activity_coverage_and_conservation(self):
        hours=[dict(hour=h,sleep=.2,work=.2,meal=.2,errand=.2,leisure=.2) for h in range(24)]
        candidate={'stage':'activity','clusters_activity':[{'cluster_id':'c1','hourly_activity':hours}]}
        check=self.ns['hard_check_activity']
        self.assertTrue(check(candidate,'00','001',clusters=[{'cluster_id':'c1'}])['ok'])
        hours[0]['sleep']=.9
        self.assertFalse(check(candidate,'00','001',clusters=[{'cluster_id':'c1'}])['ok'])
        hours.pop()
        self.assertFalse(check(candidate,'00','001',clusters=[{'cluster_id':'c1'}])['ok'])

    def test_json_retry_and_verification_without_a_provider(self):
        generator=FakeAgent('generator',['not json','{"stage":"population","clusters":[]}'])
        runner=FakeAgent('runner')
        verifier=FakeAgent('verifier',['{"should_patch":false}'])
        role=FakeAgent('discussion',['Checked.'])
        referee=FakeAgent('referee',['{"decision":"accept"}'])
        result=self.ns['run_stage_hybrid']('001','00','population',generator,'Synthetic input',runner,
            verifier,role,role,role,referee,lambda *_args,**_kw:{'ok':True,'violations':[]})
        self.assertIsNotNone(result['accepted'])
        self.assertTrue(any('valid JSON object only' in message for message in runner.sent))

    def test_hard_check_dominates_accepting_referee(self):
        generator=FakeAgent('generator',['{}'])
        runner=FakeAgent('runner')
        verifier=FakeAgent('verifier',['{"should_patch":false}'])
        role=FakeAgent('discussion',['Checked.'])
        referee=FakeAgent('referee',['{"decision":"accept"}'])
        result=self.ns['run_stage_hybrid']('001','00','population',generator,'Synthetic input',runner,
            verifier,role,role,role,referee,lambda *_args,**_kw:{'ok':False,'violations':['synthetic']},max_rounds=1)
        self.assertIsNone(result['accepted'])

    def test_agents_cannot_execute_generated_code(self):
        agents=self.ns['build_agents']({})
        self.assertIs(agents[0].kwargs['code_execution_config'],False)

    def test_original_batch_features_and_export_row_selection(self):
        with private_directory():
            frame=pd.DataFrame({'Origin':['Region 0','Region 1','Region 0','Unknown'],
                'Destination':['Region 1','Region 0','Region 0','Region 1'],
                'Trips one':[3,5,9,7],'Trips two':[2,4,8,6]})
            frame.to_csv('Flow 6-9.csv',index=False)
            planner=SimpleNamespace(decide_forces=lambda *_:{'commute':.2,'leisure':.3,'home':.4})
            engine=self.ns['HardDataEngine']('Test',SimpleNamespace(schedules={}),planner)
            engine.coords={'Region 0':np.array([0.,0.]),'Region 1':np.array([1.,0.])}
            x,y=engine._prepare(['Flow 6-9.csv'])
            self.assertEqual(x.shape,(2,8))
            np.testing.assert_array_equal(y,[5,9])
            np.testing.assert_array_equal(x[:,3],[6,6])
            self.assertEqual(engine.prepared_row_indices,[0,1])
            self.assertEqual(engine.model.n_estimators,100)
            self.assertEqual(engine.model.max_depth,5)
            self.assertEqual(engine.model.random_state,42)

    def test_small_regressor_fit_has_aligned_held_out_rows(self):
        with private_directory():
            train=pd.DataFrame({'Origin':['Region 0','Region 1','Region 0','Region 2','Region 1','Region 2']*2,
                'Destination':['Region 1','Region 0','Region 2','Region 0','Region 2','Region 1']*2,
                'flow_count':[10,10,4,4,2,2,12,12,5,5,3,3]})
            train.to_csv('Flow 6-9.csv',index=False)
            test=pd.concat([train.iloc[:2],pd.DataFrame({'Origin':['Unknown'], 'Destination':['Region 0'],'flow_count':[99]})],ignore_index=True)
            test.to_csv('Flow 22-25.csv',index=False)
            planner=SimpleNamespace(decide_forces=lambda *_:{'commute':.2,'leisure':.3,'home':.4})
            engine=self.ns['HardDataEngine']('Test',SimpleNamespace(schedules={}),planner)
            self.assertTrue(engine.learn_topology(['Flow 6-9.csv']))
            _,actual,predicted=engine.train_and_test(['Flow 6-9.csv'],'Flow 22-25.csv')
            self.assertEqual(len(actual),len(predicted))
            self.assertEqual(engine.test_row_indices,[0,1])
            self.assertTrue((predicted>=0).all())

    def test_configuration_has_no_credential_defaults(self):
        with patch.dict(os.environ,{},clear=True), patch.dict(sys.modules,{'dotenv':SimpleNamespace(load_dotenv=lambda *_a,**_k:False)}):
            with self.assertRaisesRegex(ValueError,'LLM_API_KEY'):
                load_settings(validate_inputs=False)

    def test_city_inputs_are_relative_and_missing_inputs_are_reported(self):
        with patch.dict(os.environ,{'GENURBAN_CITIES':'NYC'},clear=True), private_directory() as directory:
            cities=selected_cities()
            for name in cities['NYC']['train']+[cities['NYC']['test'],cities['NYC']['geojson']]:
                self.assertFalse(Path(name).is_absolute())
            settings=WorkflowSettings(directory,cities,[],False)
            self.assertTrue(input_errors(settings))

    def test_error_redaction(self):
        with patch.dict(os.environ,{'LLM_API_KEY':'synthetic-test-token'},clear=True):
            result=safe_error(ValueError('synthetic-test-token was rejected'))
            self.assertNotIn('synthetic-test-token',result)

    def test_model_filename_is_portable(self):
        self.assertEqual(model_filename('vendor/model:0'), 'vendor-model-0')
        self.assertEqual(model_filename('standard.model'), 'standard.model')
        self.assertEqual(model_filename('...'), 'model')
        self.assertFalse(any(char in model_filename('a<>:"/\\|?*b') for char in '<>:"/\\|?*'))


if __name__=='__main__':
    unittest.main()
