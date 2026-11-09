# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""``--enable-startup-requests`` is opt-in and refuses what it cannot serve."""

import argparse
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.utils.server_args import ServerArgs

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _args(*flags):
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    with patch.object(ServerArgs, "__post_init__", return_value=None):
        return ServerArgs.from_cli_args(parser.parse_args(["--model", "x", *flags]))


def test_startup_requests_are_off_unless_asked_for():
    assert not _args().enable_startup_requests
    args = _args("--enable-startup-requests")
    assert args.enable_startup_requests
    args.validate_startup_requests()


@pytest.mark.parametrize(
    "flags",
    [
        ("--disaggregation-mode", "prefill"),
        ("--disaggregation-mode", "decode"),
        ("--kvstore-storage-backend", "memory"),
    ],
)
def test_startup_requests_refuse_what_reaches_beyond_the_engine(flags):
    args = _args("--enable-startup-requests", *flags)
    with pytest.raises(ValueError, match="--enable-startup-requests"):
        args.validate_startup_requests()
