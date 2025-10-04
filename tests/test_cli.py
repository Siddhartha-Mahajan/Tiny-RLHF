from __future__ import annotations

from tiny_rlhf.cli.main import build_parser


def test_cli_help(capsys):
    parser = build_parser()
    parser.print_help()
    captured = capsys.readouterr()
    assert "tiny-rlhf" in captured.out
