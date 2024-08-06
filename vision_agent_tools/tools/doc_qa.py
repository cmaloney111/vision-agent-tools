import easyocr
import pytesseract
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
from vision_agent_tools.tools.shared_types import BaseTool
from vision_agent_tools.helpers.roberta_qa import RobertaQA
from io import BytesIO

class DocumentQA(BaseTool):
    """
    A tool to extract text from images or PDF files and answer questions based on the extracted text.
    """

    def __init__(self):
        """
        Initializes the DocumentQA tool with an OCR tool and a QA model.
        """
        self.ocr_tool_image = easyocr.Reader(['en'])
        self.ocr_tool_pdf = pytesseract
        self._roberta_qa = RobertaQA()

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extracts text from an image using EasyOCR.

        Args:
            image (Image.Image): The input image containing a document.

        Returns:
            str: The extracted text from the image.
        """
        image_np = np.array(image)
        results = self.ocr_tool_image.readtext(image_np)
        text = " ".join([result[1] for result in results])
        return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file by converting each page to images and then applying Tesseract OCR.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: The extracted text from the PDF.
        """
        text = ""
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(BytesIO(pix.tobytes()))
            text += self.ocr_tool_pdf.image_to_string(img)
        return text

    def answer_question(self, text: str, question: str) -> str:
        """
        Answers a question based on the provided text using a QA model.

        Args:
            text (str): The text extracted from the document.
            question (str): The question to be answered.

        Returns:
            str: The answer to the question.
        """
        result = self._roberta_qa(context=text, question=question)
        return result.answer

    def __call__(self, file, question: str) -> str:
        """
        Extracts text from an image or PDF file and answers a question based on the extracted text.

        Args:
            file: The file to be analyzed (can be an image or PDF).
            question (str): The question to be answered.

        Returns:
            str: The answer to the question.
        """
        if isinstance(file, Image.Image):
            text = self.extract_text_from_image(file)
        elif isinstance(file, str) and file.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file)
        else:
            raise TypeError("Unsupported file type. Provide an image or a PDF file.")

        return self.answer_question(text, question)
