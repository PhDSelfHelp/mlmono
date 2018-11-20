import unittest

from ml.base import GlobalConfig, MLConfig
from ml.base.cli_utils import override_global_config


class TestCliUtils(unittest.TestCase):

    SAMPLE_CONFIG_PATH = "ml/base/tests/sample_config.yaml"

    def setUp(self):
        # TODO(jdaaph): Change to relative file path import.
        self.global_config = GlobalConfig.from_file(TestCliUtils.SAMPLE_CONFIG_PATH)

    def test_override_global_config(self):
        custom_args = ["mode=eval"]
        override_global_config(self.global_config,
                               custom_args,
                               can_create=False)

        self.assertEqual(self.global_config.mode, "eval")

    def test_override_global_config_parsing_order(self):
        custom_args = ["mode=0"]
        override_global_config(self.global_config,
                               custom_args,
                               can_create=False)

        self.assertEqual(self.global_config.mode, 0)
        self.assertEqual(type(self.global_config.mode), int)

        custom_args = ["mode=0."]
        override_global_config(self.global_config,
                               custom_args,
                               can_create=False)

        self.assertAlmostEqual(self.global_config.mode, 0)
        self.assertEqual(type(self.global_config.mode), float)

        custom_args = ["mode=0.x"]
        override_global_config(self.global_config,
                               custom_args,
                               can_create=False)

        self.assertAlmostEqual(self.global_config.mode, "0.x")
        self.assertEqual(type(self.global_config.mode), str)

    def test_override_global_config_create(self):
        custom_args = ["io__created_int=1"]
        override_global_config(self.global_config,
                               custom_args,
                               can_create=True)

        self.assertAlmostEqual(self.global_config.io.created_int, 1)
        self.assertEqual(type(self.global_config.io.created_int), int)

    def test_override_global_config_composite(self):
        custom_args = ["io__created_int=1", "mode=0.x"]
        override_global_config(self.global_config,
                               custom_args,
                               can_create=True)

        self.assertAlmostEqual(self.global_config.io.created_int, 1)
        self.assertEqual(type(self.global_config.io.created_int), int)

        self.assertAlmostEqual(self.global_config.mode, "0.x")
        self.assertEqual(type(self.global_config.mode), str)


    def test_override_global_config_raises(self):
        with self.assertRaises(Exception) as context:
            custom_args = ["io__io_name__nonexist=10"]
            override_global_config(self.global_config,
                                   custom_args,
                                   can_create=False)

        with self.assertRaises(Exception) as context:
            custom_args = ["io=eval"]
            override_global_config(self.global_config,
                                   custom_args,
                                   can_create=False)
