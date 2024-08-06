from PIL import Image
from vision_agent_tools.tools.doc_qa import DocumentQA


def test_successful_doc_qa_from_image():
    test_image = "palm_oil.png"

    image = Image.open(f"tests/tools/data/doc_qa/{test_image}")

    document_qa = DocumentQA()

    answer = document_qa(file=image, question="what is the most efficient vegetable oil?")

    assert answer == "Palm oil"

def test_successful_doc_qa_from_pdf():
    test_pdf = "animal_bytes.pdf"

    file = f"tests/tools/data/doc_qa/{test_pdf}"

    document_qa = DocumentQA()

    answer = document_qa(file=file, question="What is the percentage range of attribute LAIs that are due to unknown incidents?")

    assert answer == "6% and 74%"

test_successful_doc_qa_from_pdf()
test_successful_doc_qa_from_image()