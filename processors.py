from __future__ import absolute_import

import os
from abc import ABC

try:
    from model_utils import readfile, read_jsonl_file, get_dataset_examples
except ImportError:
    from .model_utils import readfile, read_jsonl_file, get_dataset_examples


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a=None, text_b=None, label=None, image=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.image = image


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class DataProcessorNLP(DataProcessor, ABC):
    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        return readfile(input_file)


class BaseProcessorNLP(DataProcessorNLP):
    def get_train_examples(self, data_dir, file_name="train.txt"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name="valid.txt"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "dev")

    def get_test_examples(self, data_dir, file_name="test.txt"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), "test")

    def get_labels(self):
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ConllNerProcessor(BaseProcessorNLP):
    """Processor for the CoNLL-2003 data set."""
    def get_labels(self):
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]


class JNLNerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B-DNA", "I-DNA", "B-RNA", "I-RNA", "B-protein", "I-protein", "B-cell_type", "I-cell_type",
                "B-cell_line", "I-cell_line", "[CLS]", "[SEP]"]

class WNUT17NerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B-location", "I-location", "B-group", "I-group", "B-corporation", "I-corporation", "B-person", "I-person", "B-product",
                "I-product", "B-creative-work", "I-creative-work", "[CLS]", "[SEP]"]


class SimplifiedNerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B", "I", "[CLS]", "[SEP]"]


class BC2NerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B-GENE", "I-GENE", "[CLS]", "[SEP]"]


class BC4NerProcessor(BaseProcessorNLP):
    def get_labels(self):
        return ["O", "B-Chemical", "I-Chemical", "[CLS]", "[SEP]"]


class GedProcessor(BaseProcessorNLP):
    """ Processor for GED TSV data """

    def get_labels(self):
        return ["c", "i", "[CLS]", "[SEP]"]


class BaseNLIProcessor(BaseProcessorNLP):
    def get_train_examples(self, data_dir, file_name="train.jsonl"):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, file_name)), "train")

    def get_dev_examples(self, data_dir, file_name="multinli_1.0_dev_matched.jsonl"):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, file_name)), "dev")

    def get_test_examples(self, data_dir, file_name="multinli_1.0_dev_mismatched.jsonl"):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, file_name)), "test")

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence1, sentence2, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=sentence1, text_b=sentence2, label=label))
        return examples

    @classmethod
    def _read_jsonl(cls, input_file):
        return read_jsonl_file(input_file)


class MNLIProcessor(BaseNLIProcessor):
    def get_labels(self):
        return ["neutral", "entailment", "contradiction"]


class HANSProcessor(BaseNLIProcessor):
    def get_labels(self):
        return ["entailment", "non-entailment"]

    def get_dev_examples(self, data_dir, file_name="valid.jsonl"):
        return super().get_dev_examples(data_dir, file_name)

    def get_test_examples(self, data_dir, file_name="valid.jsonl"):
        return super().get_test_examples(data_dir, file_name)


class BaseProcessorImageClassification(DataProcessor):
    dataset_name = ""
    def get_test_examples(self, data_dir):
        return self._create_examples(self._get_dataset_examples("test"), "test")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._get_dataset_examples("dev"), "dev")

    def get_train_examples(self, data_dir):
        return self._create_examples(self._get_dataset_examples("train"), "train")

    def get_labels(self):
        raise NotImplementedError()

    def _create_examples(self, data, set_type):
        examples = []
        for i, (image, label) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, image=image, label=label))
        return examples

    @classmethod
    def _get_dataset_examples(cls, task="train"):
        return get_dataset_examples(cls.dataset_name, task)


class MNISTProcessor(BaseProcessorImageClassification):
    dataset_name = "MNIST"
    def get_labels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class CIFAR10Processor(BaseProcessorImageClassification):
    dataset_name = "CIFAR10"
    def get_labels(self):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class CIFAR100Processor(BaseProcessorImageClassification):
    dataset_name = "CIFAR100"
    def get_labels(self):
        return list(range(100))