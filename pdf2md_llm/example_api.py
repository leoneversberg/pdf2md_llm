import sys
import os
from vllm import SamplingParams
from pdf2md_llm.llm import llm_model
from pdf2md_llm.pdf2img import PdfToImg

if __name__ == "__main__":
    pdf_files = ["example.pdf"]

    for pdf_file in pdf_files:
        pdf2img = PdfToImg(size=(700, None), output_folder="./out")
        img_files = pdf2img.convert(pdf_file)

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

        # Extract the filename without extension and use it for the output file
        base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
        output_file = f"{base_filename}.md"
        output_path = os.path.join(pdf2img.output_folder, output_file)

        # Delete the output file if it already exists
        if os.path.exists(output_file):
            os.remove(output_file)

        # Append all pages to one Markdown file
        for img_file in img_files:
            markdown_text = llm.generate(
                img_file, sampling_params=sampling_params
            )  # convert image to Markdown with LLM
            with open(output_path, "a", encoding="utf-8") as myfile:
                myfile.write(markdown_text)
