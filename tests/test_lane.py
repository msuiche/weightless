import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lane as L


class LaneFileTests(unittest.TestCase):
    def _lane_dir(self, td, name, doc):
        d = os.path.join(td, name)
        os.makedirs(d)
        with open(os.path.join(d, "weightless.json"), "w") as f:
            json.dump(doc, f)
        return d

    def test_load_lane_schema_gate(self):
        with tempfile.TemporaryDirectory() as td:
            self._lane_dir(td, "ok", {"schema": L.SCHEMA, "name": "ok"})
            old = L.LANES_DIR
            L.LANES_DIR = td
            try:
                self.assertEqual(L.load_lane("ok")["name"], "ok")
            finally:
                L.LANES_DIR = old

    def test_local_override_merges_recursively(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._lane_dir(td, "ok", {
                "schema": L.SCHEMA, "name": "ok",
                "pin": {"ref": "main", "note": "committed"},
                "checkout": "/committed/path"})
            with open(os.path.join(d, "weightless.local.json"), "w") as f:
                json.dump({"pin": {"ref": "fix-branch"},
                           "checkout": "/local/path"}, f)
            old = L.LANES_DIR
            L.LANES_DIR = td
            try:
                lane = L.load_lane("ok")
            finally:
                L.LANES_DIR = old
            self.assertEqual(lane["pin"]["ref"], "fix-branch")
            self.assertEqual(lane["pin"]["note"], "committed")
            self.assertEqual(lane["checkout"], "/local/path")

    def test_load_lane_refuses_wrong_schema(self):
        with tempfile.TemporaryDirectory() as td:
            self._lane_dir(td, "bad", {"schema": "weightless/lane/0", "name": "bad"})
            old = L.LANES_DIR
            L.LANES_DIR = td
            try:
                with self.assertRaises(SystemExit):
                    L.load_lane("bad")
            finally:
                L.LANES_DIR = old

    def test_load_lane_missing_refuses(self):
        old = L.LANES_DIR
        L.LANES_DIR = tempfile.mkdtemp()
        try:
            with self.assertRaises(SystemExit):
                L.load_lane("nope")
        finally:
            L.LANES_DIR = old


class ShaTests(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(b"abc")
            p = f.name
        self.assertEqual(L.sha256_file(p),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        os.unlink(p)

    def test_verify_vector_mismatch_refuses(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(b"abc")
            p = f.name
        with self.assertRaises(SystemExit):
            L.verify_vector(p, "0" * 64)
        os.unlink(p)

    def test_verify_vector_ok(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(b"abc")
            p = f.name
        self.assertEqual(L.verify_vector(p, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        os.unlink(p)


class ConfigTests(unittest.TestCase):
    def _repo(self, td, cfg_doc):
        cfg_path = os.path.join(td, "config")
        os.makedirs(cfg_path)
        with open(os.path.join(cfg_path, "cluster.json"), "w") as f:
            json.dump(cfg_doc, f)

    def test_lane_config_vector_reads_fields(self):
        with tempfile.TemporaryDirectory() as td:
            self._repo(td, {"glp": {"repo": "org/vec", "revision": "a" * 40,
                                    "file": "v.gguf", "sha256": "b" * 64,
                                    "served_model_name": "m-weightless"}})
            lane = {"glp": {"config": {"path": "config/cluster.json", "key": "glp",
                                       "fields": {"repo": "repo", "revision": "revision",
                                                  "file": "file", "sha256": "sha256"}}}}
            got = L.lane_config_vector(td, lane)
            self.assertEqual(got, ("org/vec", "a" * 40, "v.gguf", "b" * 64))

    def test_lane_config_vector_incomplete_is_none(self):
        with tempfile.TemporaryDirectory() as td:
            self._repo(td, {"glp": {"repo": "org/vec"}})
            lane = {"glp": {"config": {"path": "config/cluster.json", "key": "glp",
                                       "fields": {}}}}
            self.assertIsNone(L.lane_config_vector(td, lane))

    def test_env_for(self):
        lane = {"glp": {"env": {"path": "GLP_FILE", "alpha": "GLP_ALPHA",
                                "extra": {"NO_DRAFTS": "1"}}}}
        self.assertEqual(L.env_for(lane, "/v/x.gguf", 3.5),
                         {"GLP_FILE": "/v/x.gguf", "GLP_ALPHA": "3.5", "NO_DRAFTS": "1"})
        self.assertEqual(L.env_for(lane, None, None),
                         {"GLP_FILE": "", "NO_DRAFTS": "1"})


if __name__ == "__main__":
    unittest.main()
