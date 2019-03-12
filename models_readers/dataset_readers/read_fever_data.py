
from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField,MetadataField,Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import sys
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("fever")
class FeverDatasetReader(DatasetReader):
    """
    Reads a JSON-lines file containing papers from the Semantic Scholar database, and creates a
    dataset suitable for document classification using these papers.

    Expected format for each input line: {"paperAbstract": "text", "title": "text", "venue": "text"}

    The JSON could have other fields, too, but they are ignored.

    The output of ``read`` is a list of ``Instance`` s with the fields:
        title: ``TextField``
        abstract: ``TextField``
        label: ``LabelField``

    where the ``label`` is derived from the venue of the paper.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,lazy : bool = True,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def truncate(self, sent):

        ## truncate at 1000 words. irrespective of claim or evidence truncate it at n...
        # Else it was overloading memory due to the packing/padding of all sentences into the longest size..
        # which was like 180k words or something
        #todo: make 1000 come from command line arguments
        tr_len=1000

        sent_split = sent.split(" ")
        if (len(sent_split) > tr_len):
            sent_tr = sent_split[:1000]
            sent2 = " ".join(sent_tr)
            return sent2
        else:
            return sent

    @overrides
    def _read(self, file_path):
        with open((file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                claim = paper_json['claim']
                claim= self.truncate(claim)
                evidence_list = paper_json['sents']
                evidence=" ".join(evidence_list)
                evidence = self.truncate(evidence)
                gold_label = paper_json['label']
                yield self.text_to_instance(claim, evidence, gold_label)
        logger.info("done reading")

    @overrides
    def text_to_instance(self, claim: str, evidence: str, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        logger.debug(f"value of claim is:{claim}")
        logger.debug(f"value of evidence is:{evidence}")
        logger.debug(f"value of label is:{label}")

        claims_tokens = self._tokenizer.tokenize(claim)
        evidence_tokens = self._tokenizer.tokenize(evidence)
        title_field = TextField(claims_tokens, self._token_indexers)
        abstract_field = TextField(evidence_tokens, self._token_indexers)
        fields = {'claim': title_field, 'evidence': abstract_field}
        if label is not None:
            fields['label'] = LabelField(label)

        metadata = {"claims_tokens": [x.text for x in claims_tokens],
                    "evidence_tokens": [x.text for x in evidence_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)