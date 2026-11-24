#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model files are located without importing `face_recognition_models`,
so `face_recognition` works even when `pkg_resources` is unavailable
(setuptools>=82)."""

import os
import subprocess
import sys
import textwrap
import unittest

from face_recognition import _models


def run_snippet(code):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True)


class Test_models_location(unittest.TestCase):

    def test_all_model_files_exist(self):
        for locate in (_models.pose_predictor_model_location,
                       _models.pose_predictor_five_point_model_location,
                       _models.face_recognition_model_location,
                       _models.cnn_face_detector_model_location):
            self.assertTrue(os.path.isfile(locate()), locate())

    def test_import_without_pkg_resources(self):
        result = run_snippet("""
            import sys
            sys.modules["pkg_resources"] = None  # as on setuptools>=82
            import face_recognition
            print("OK")
        """)
        self.assertIn("OK", result.stdout, result.stdout + result.stderr)

    def test_missing_models_raises_import_error(self):
        result = run_snippet("""
            import sys
            sys.modules["face_recognition_models"] = None
            try:
                import face_recognition
            except ImportError as e:
                assert "pip install face_recognition_models" in str(e), str(e)
                print("IMPORT_ERROR_RAISED")
        """)
        self.assertIn("IMPORT_ERROR_RAISED", result.stdout, result.stdout + result.stderr)

    def test_missing_models_exits_with_failure_status(self):
        result = run_snippet("""
            import sys
            sys.modules["face_recognition_models"] = None
            import face_recognition
        """)
        self.assertNotEqual(result.returncode, 0)
