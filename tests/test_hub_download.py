"""Offline regression checks for public Hugging Face access policy.

No model download, Torch, Transformers or HF login is required to run these.
"""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "gpt2_gate.py"
TEXT = SOURCE.read_text(encoding="utf-8")
TREE = ast.parse(TEXT, filename=str(SOURCE))


class PublicHubDownloadTests(unittest.TestCase):
    def test_disable_implicit_auth_before_third_party_imports(self):
        # Important: huggingface_hub snapshots this flag at import time.
        flag = 'os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"'
        self.assertIn(flag, TEXT)
        self.assertLess(TEXT.index(flag), TEXT.index("import torch"))
        self.assertLess(TEXT.index(flag), TEXT.index("import numpy"))

    def test_token_opt_in_function(self):
        node = next(
            n for n in TREE.body
            if isinstance(n, ast.FunctionDef) and n.name == "hub_token_arg"
        )
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        fn = namespace["hub_token_arg"]
        self.assertIs(fn(False), False)
        self.assertIs(fn(True), True)

    def test_both_model_files_and_dataset_explicitly_opt_out(self):
        expected = {"AutoTokenizer.from_pretrained", "GPT2LMHeadModel.from_pretrained", "load_dataset"}
        seen = set()
        for node in ast.walk(TREE):
            if not isinstance(node, ast.Call):
                continue
            name = (
                ast.unparse(node.func)
                if hasattr(ast, "unparse")
                else ""
            )
            if name not in expected:
                continue
            kws = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
            self.assertEqual(kws.get("token"), "hub_token_arg(args.hf_auth)", name)
            seen.add(name)
        self.assertEqual(seen, expected)
        self.assertIn('ap.add_argument("--hf-auth"', TEXT)


if __name__ == "__main__":
    unittest.main()
