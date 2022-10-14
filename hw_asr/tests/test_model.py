import unittest

from tqdm import tqdm

from hw_asr.collate_fn.collate import collate_fn
from hw_asr.datasets import LibrispeechDataset
from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.model.deepspeech2 import DeepSpeech2


class TestDataloader(unittest.TestCase):
    def test_collate_fn(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            ds = LibrispeechDataset(
                "dev-clean", text_encoder=config_parser.get_text_encoder(),
                config_parser=config_parser
            )

            batch_size = 3
            batch = collate_fn([ds[i] for i in range(batch_size)])

            model = DeepSpeech2(128, 33)
            model(**batch)