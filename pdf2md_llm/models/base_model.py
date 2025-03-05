from abc import ABC, abstractmethod
from vllm import SamplingParams


def _post_process_text(text: str) -> str:
    """
    Process the input text by stripping markdown code block delimiters
    and appending newlines for multiple outputs.

    Args:
        text (str): The input text to process.

    Returns:
        str: The processed text.
    """
    if text.startswith("```markdown\n"):
        text = text.lstrip("```markdown\n")
        if text.endswith("\n```"):
            text = text.rstrip("\n```")

    text += "\n\n"  # add newlines so that multiple outputs can be appended
    return text


class BaseModel(ABC):
    """
    Abstract base class for language models.

    This class defines the common interface for all language models used in the
    pdf2md_llm package. Subclasses must implement the `generate` method.

    Attributes:
        model (str): The name or path of the model.
    """

    def __init__(self, model: str, **kwargs):
        """
        Initialize the BaseModel.

        Args:
            model (str): The name or path of the model.
            **kwargs: Additional keyword arguments for model configuration.
        """
        self.model = model

    @abstractmethod
    def generate(
        self, file_path: str, prompt: str, sampling_params: SamplingParams
    ) -> str:
        """
        Generate text from an input file using the model.

        This method must be implemented by subclasses.

        Args:
            file_path (str): The path to the input file (e.g., an image of a PDF page).
            prompt (str): The prompt to guide the text generation.
            sampling_params (SamplingParams): The parameters for sampling during text generation.

        Returns:
            str: The generated text.
        """
        pass
