"""Offline checks for the CodeBuild → exact-image Modal preparation handoff."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

# CodeBuild mounts infra beside the installed library, outside sys.path.
spec = importlib.util.spec_from_file_location(
    "rebuild_modal_image_cli", Path(__file__).resolve().parents[1] / "scripts/rebuild_modal_image.py"
)
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)


class RebuildModalImageTests(unittest.TestCase):
    def invoke(self, *arguments):
        output = io.StringIO()
        with patch("sys.argv", ["rebuild_modal_image.py", *arguments]), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = rebuild.main()
        return code, output.getvalue()

    def test_dry_run_executes_no_commands(self):
        with patch.object(rebuild.subprocess, "Popen") as popen, \
                patch.object(rebuild.subprocess, "run") as run:
            code, output = self.invoke("--dry-run")
        self.assertEqual(code, 0)
        popen.assert_not_called()
        run.assert_not_called()
        self.assertIn("infra/scripts/build_image.py", output)
        self.assertIn("infra.modal.prepare_image", output)

    def run_flow(self, *, fail=None, build_output=None):
        calls = []
        repository = "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom-runtime"

        def execute(command, *, env, **kwargs):
            calls.append((command, dict(env)))
            if fail and fail in command:
                raise subprocess.CalledProcessError(1, command)
            output = ""
            if command[:3] == ["git", "rev-parse", "--short"]:
                output = "abcdef0\n"
            elif "describe-repositories" in command:
                output = repository + "\n"
            elif "infra/scripts/build_image.py" in command:
                output = build_output if build_output is not None else repository + ":release\n"
            return subprocess.CompletedProcess(command, 0, stdout=output)

        with patch.object(rebuild, "execute", side_effect=execute), \
                patch.dict(os.environ, {"CS182_MODAL_IMAGE": "old-image"}):
            code, output = self.invoke("--image-tag", "release", "--modal-profile", "test-profile")
        return code, output, calls, repository + ":release"

    def test_prepares_exact_build_result_with_selected_profile(self):
        code, _, calls, image = self.run_flow()
        self.assertEqual(code, 0)
        command, env = calls[-1]
        self.assertEqual(command[-3:], ["python", "-m", "infra.modal.prepare_image"])
        self.assertEqual(env["CS182_MODAL_IMAGE"], image)
        self.assertEqual(env["MODAL_PROFILE"], "test-profile")
        self.assertFalse(any("discover_modal.py" in command for command, _ in calls))

    def test_build_failure_never_prepares(self):
        code, _, calls, _ = self.run_flow(fail="infra/scripts/build_image.py")
        self.assertEqual(code, 1)
        self.assertFalse(any("infra.modal.prepare_image" in command for command, _ in calls))

    def test_lockfile_failure_stops_before_aws(self):
        code, _, calls, _ = self.run_flow(fail="--check")
        self.assertEqual(code, 1)
        self.assertFalse(any(command[0] == "aws" for command, _ in calls))
        checked = [command for command, _ in calls if command[:2] == ["uv", "lock"]]
        self.assertEqual(len(checked), 1)

    def test_unexpected_build_uri_never_prepares(self):
        code, _, calls, _ = self.run_flow(build_output="another-image:release\n")
        self.assertEqual(code, 1)
        self.assertFalse(any("infra.modal.prepare_image" in command for command, _ in calls))

    def test_preparation_failure_prints_retry_for_published_image(self):
        code, output, _, image = self.run_flow(fail="infra.modal.prepare_image")
        self.assertEqual(code, 1)
        self.assertIn("without rebuilding", output)
        self.assertIn("export CS182_MODAL_IMAGE=" + image, output)
        self.assertIn("export MODAL_PROFILE=test-profile", output)


if __name__ == "__main__":
    unittest.main()
