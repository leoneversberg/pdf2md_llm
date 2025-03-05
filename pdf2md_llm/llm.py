from pdf2md_llm.models.qwen import Qwen25VLModel
from pdf2md_llm.models.base_model import BaseModel


def llm_model(model: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ", **kwargs) -> BaseModel:
    """
    Factory function to create an instance of a language model.

    This function returns an instance of a language model based on the provided model name.
    Currently, it supports the Qwen2.5-VL model.

    Args:
        model (str): The name or path of the model. Default is "Qwen/Qwen2.5-VL-3B-Instruct-AWQ".
        **kwargs: Additional keyword arguments for model configuration.

    Returns:
        BaseModel: An instance of a subclass of BaseModel corresponding to the specified model.

    Raises:
        ValueError: If the specified model is not supported.
    """
    if "Qwen2.5-VL" in model:
        return Qwen25VLModel(model=model, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model}")
