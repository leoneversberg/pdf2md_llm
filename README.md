# pdf2md_llm

`pdf2md_llm` is a Python package that converts PDF files to Markdown using a local Large Language Model (LLM). 

The package leverages the `pdf2image` library to convert PDF pages to images and a vision language model to generate Markdown text from these images.

## Features

- Convert PDF files to images.
- Generate Markdown text from images using a local LLM.
- Keep your data private. No third-party file uploads. 

## Installation

You need a CUDA compatible GPU to run local LLMs with vLLM.

You can use `pip` to install the package:

```bash
pip install pdf2md-llm
```
## Usage

### CLI

You can use the `pdf2md_llm` package via the **command line interface (CLI)**.

To convert a PDF file to Markdown, run the following command:

```bash
pdf2md_llm <pdf_file> [options]
```

#### Options

* `pdf_file`: Path to the PDF file to convert.
* `--model`: Name of the model to use (default: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`).
* `--dtype`: Data type for the model weights and activations (default: `None`).
* `--max_model_len`: Max model context length (default: `7000`).
* `--size`: Image size as a tuple (default: `(700, None)`).
* `--dpi`: DPI of the images (default: `200`).
* `--fmt`: Image format (default: `jpeg`).
* `--output_folder`: Folder to save the output Markdown file (default: `./out`).

#### Example

```bash
pdf2md_llm example.pdf --model "Qwen/Qwen2.5-VL-3B-Instruct-AWQ" --output_folder "./output"
```

##### Model Support:
Currently the following Qwen2.5-VL models are supported: 

* `Qwen/Qwen2.5-VL-3B-Instruct`
* `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`
* `Qwen/Qwen2.5-VL-7B-Instruct`
* `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
* `Qwen/Qwen2.5-VL-72B-Instruct`
* `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`

If you want to use a different model, feel free to add a vLLM compatible model to the factory function `llm_model()` in `llm.py`

### Python API

You can use the `pdf2md_llm` package via the **Python API**.

Basic usage:

```python
from vllm import SamplingParams

from pdf2md_llm.llm import llm_model
from pdf2md_llm.pdf2img import PdfToImg

pdf2img = PdfToImg(size=(700, None), output_folder="./out")
img_files = pdf2img.convert("example.pdf")

llm = llm_model(
    model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    dtype="half",
    max_num_seqs=1,
    max_model_len=7000,
)

sampling_params = SamplingParams(
    temperature=0.1,
    min_p=0.1,
    max_tokens=8192,
    stop_token_ids=[],
)

# Append all pages to one output Markdown file
for img_file in img_files:
    markdown_text = llm.generate(
        img_file, sampling_params=sampling_params
    )  # convert image to Markdown with LLM
    with open("example.md", "a", encoding="utf-8") as myfile:
        myfile.write(markdown_text)
```

For a full example, see [example_api.py](./pdf2md_llm/example_api.py)


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

* [pdf2image](https://github.com/Belval/pdf2image) for converting PDF files to images.

* [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) LLM model

* [vLLM](https://github.com/vllm-project/vllm) for efficient LLM model inference