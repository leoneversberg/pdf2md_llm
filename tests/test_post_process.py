import sys
import os
import pytest

from pdf2md_llm.models.base_model import _post_process_text


def test_post_process_text_with_markdown_delimiters():
    input_text = "```markdown\nThis is a test.\n```"
    expected_output = "This is a test.\n\n"
    assert _post_process_text(input_text) == expected_output


def test_post_process_text_without_markdown_delimiters():
    input_text = "This is a test."
    expected_output = "This is a test.\n\n"
    assert _post_process_text(input_text) == expected_output


def test_post_process_text_with_partial_markdown_delimiters():
    input_text = "```markdown\nThis is a test."
    expected_output = "This is a test.\n\n"
    assert _post_process_text(input_text) == expected_output


def test_post_process_text_with_no_text():
    input_text = ""
    expected_output = "\n\n"
    assert _post_process_text(input_text) == expected_output


def test_post_process_text_with_code():
    input_text = "Test.\n```python\nprint('Hello')\n```"
    expected_output = "Test.\n```python\nprint('Hello')\n```\n\n"
    assert _post_process_text(input_text) == expected_output
