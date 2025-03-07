import argparse
import os
import sys
from pdf2md_llm.llm import llm_model
from pdf2md_llm.pdf2img import PdfToImg


def validate_file_path(file_path):
    if not os.path.isfile(file_path):
        raise argparse.ArgumentTypeError(f"File {file_path} does not exist.")
    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using a local LLM."
    )
    parser.add_argument(
        "pdf_file", type=validate_file_path, help="Path to the PDF file to convert."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type for the model weights and activations.",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"]
    )
    parser.add_argument(
        "--max_model_len", type=int, default=7000, help="Max model context length."
    )
    parser.add_argument("--size", type=tuple, default=(700, None), help="Image size.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI of the images.")
    parser.add_argument("--fmt", type=str, default="jpeg", help="Image format.", choices=["jpeg", "png", "ppm"])
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./out",
        help="Folder to save the output Markdown file.",
    )
    args = parser.parse_args()

    try:
        # convert PDF to images
        print(f"Converting {args.pdf_file} to images ...")
        pdf2img = PdfToImg(
            size=args.size, dpi=args.dpi, fmt=args.fmt, output_folder=args.output_folder
        )
        img_files = pdf2img.convert(args.pdf_file)
    except Exception as e:
        print(f"Error converting PDF {args.pdf_file} to images: {e}")
        sys.exit(1)

    # load the LLM model
    dtype = (
        args.dtype
        if args.dtype
        else ("half" if "awq" in args.model.lower() else "auto")
    )  # "auto" is not supported for AWQ quantization

    try:
        llm = llm_model(
            model=args.model,
            dtype=dtype,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
        )
    except Exception as e:
        print(f"Error loading the LLM model {args.model}: {e}")
        sys.exit(1)

    # Extract the filename without extension and use it for the output file
    base_filename = os.path.splitext(os.path.basename(args.pdf_file))[0]
    output_file = f"{base_filename}.md"
    output_path = os.path.join(pdf2img.output_folder, output_file)

    # Delete the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Append all pages to one Markdown file
    for img_file in img_files:
        try:
            markdown_text = llm.generate(img_file)  # convert image to Markdown with LLM
        except Exception as e:
            print(f"Error generating Markdown from image {img_file}: {e}")
            continue
        with open(output_path, "a", encoding="utf-8") as myfile:
            myfile.write(markdown_text)

    print(f"Markdown file saved to: {output_path}")


if __name__ == "__main__":
    main()
