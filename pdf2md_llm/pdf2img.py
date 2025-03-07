import os
import shutil
from pdf2image import convert_from_path


class PdfToImg:
    """
    A class to convert PDF files to images.

    This class provides functionality to convert PDF files to images using the pdf2image library.

    Attributes:
        dpi (int): The DPI (dots per inch) of the resulting images. Default is 200.
        fmt (str): The format of the resulting images. Default is "jpeg".
        size (tuple): The size (width, height) of the resulting images. Default is (700, None).
        output_folder (str): The folder to save the output images. Default is "./out".
    """

    def __init__(
        self,
        dpi: int = 200,
        fmt: str = "jpeg",
        size: tuple = (700, None),
        output_folder: str = "./out",
    ):
        """
        Initialize the PdfToImg class.

        Args:
            dpi (int): The DPI (dots per inch) of the resulting images. Default is 200.
            fmt (str): The format of the resulting images. Default is "jpeg".
            size (tuple): The size (width, height) of the resulting images. Default is (700, None).
            output_folder (str): The folder to save the output images. Default is "./out".
        """
        self.fmt = fmt
        self.output_folder = output_folder
        self.paths_only = True
        self.size = size  # shape of the resulting images
        self.dpi = dpi  # dpi of the resulting images

        # Remove the output folder if it already exists
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

        os.makedirs(self.output_folder, exist_ok=True)

    def convert(self, file_path: str, **kwargs) -> list[str]:
        """
        Convert a PDF file to images.

        This method converts a PDF file to images and saves them in the specified output folder.

        Args:
            file_path (str): The path to the PDF file to convert.
            **kwargs: Additional keyword arguments for the pdf2image.convert_from_path function.

        Returns:
            list[str]: A list of paths to the generated image files.
        """

        img_paths = convert_from_path(
            file_path,
            fmt=self.fmt,
            output_folder=self.output_folder,
            paths_only=self.paths_only,
            size=self.size,
            dpi=self.dpi,
            **kwargs,
        )
        
        if img_paths is None or len(img_paths) == 0:
            raise ValueError("No images generated.")

        return img_paths
