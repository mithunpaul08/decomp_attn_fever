# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from models_readers.dataset_readers.read_fever_data import FeverDatasetReader



class TestFeverDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = FeverDatasetReader()
        instances = ensure_list(reader.read('tests/fixtures/fever_train_fixture.jsonl'))


        instance1 = {"title": ["Nikolaj", "Coster","-","Waldau", "worked"],
                     "abstract": ["In","2017",",","he","became"],
                     "venue": "SUPPORTS"}


        assert len(instances) == 10

        fields = instances[0].fields
        assert [t.text for t in fields["title"].tokens[:5]] == instance1["title"]
        assert [t.text for t in fields["abstract"].tokens[:5]] == instance1["abstract"]
        assert fields["label"].label == instance1["venue"]


