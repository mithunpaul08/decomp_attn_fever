# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class DecomposableAttentionClassifierTest(ModelTestCase):
    def setUp(self):
        super(DecomposableAttentionClassifierTest, self).setUp()
        self.set_up_model('tests/fixtures/decomp_attn_classifier.json',
                          'tests/fixtures/fever_train_fixture.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)