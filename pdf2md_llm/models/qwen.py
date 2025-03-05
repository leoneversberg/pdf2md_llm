from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from pdf2md_llm.models.base_model import BaseModel, _post_process_text


class Qwen25VLModel(BaseModel):
    """
    Qwen25VLModel is a subclass of BaseModel that implements the generate method
    for the Qwen2.5-VL model.

    Attributes:
        model (str): The name or path of the model.
        llm (LLM): The language model instance.
    """

    def __init__(self, model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", **kwargs):
        """
        Initialize the Qwen25VLModel.

        Args:
            model (str): The name or path of the model. Default is "Qwen/Qwen2.5-VL-3B-Instruct-AWQ".
            **kwargs: Additional keyword arguments for the vLLM LLM class configuration.
        """
        super().__init__(model, **kwargs)
        self.llm = LLM(
            model=self.model, limit_mm_per_prompt={"image": 1, "video": 0}, **kwargs
        )

    def generate(
        self, file_path: str, prompt: str = None, sampling_params: SamplingParams = None
    ) -> str:
        """
        Generate text from an input file using the Qwen2.5-VL model.

        Args:
            file_path (str): The path to the input image file (e.g., an image of a PDF page).
            prompt (str): The prompt to guide the text generation. Default is a prompt to convert the image to Markdown.
            sampling_params (SamplingParams): The vLLM parameters for sampling during text generation. Default is a set of predefined parameters.

        Returns:
            str: The generated text.
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.1,
                min_p=0.1,
                max_tokens=8192,
                stop_token_ids=[],
            )
        if prompt is None:
            prompt = "Convert the provided image of a PDF document strictly into valid Markdown. Do not add additional text that is not in the image!"

        message = [
            {"role": "system", "content": "You are a tool to parse documents."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{file_path}",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        print(f"Parsing {file_path} ...")

        processor = AutoProcessor.from_pretrained(self.model)
        prompt = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_input, _ = process_vision_info(message)

        mm_data = {}
        if image_input is not None:
            mm_data["image"] = image_input

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text

        return _post_process_text(generated_text)
