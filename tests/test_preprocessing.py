import pytest
from rag.preprocessing import Preprocessing

@pytest.fixture
def preprocessing():
    return Preprocessing.__new__(Preprocessing)

class FakeOCR:
    def predict(self, img):
        return [
            {
                "rec_texts": ["Fake OCR text"]
            }
        ]

def test_process_pdf():
    preprocessing = Preprocessing(ocr = FakeOCR())
    result = preprocessing.process_pdf("tests/data/test.pdf")
    assert len(result) > 0
    assert len(result) == 6
    

def test_preprocess_text(preprocessing):
    pages = [
        "Ｔｅｓｔ Ｔｅｘｔ",
        "Hello\u200bWorld",
        "Normal text",
        "ＡＢＣ １２３",
    ]
    result = preprocessing.clean_text(pages)
    assert result == [
        "Test Text",
        "HelloWorld",
        "Normal text",
        "ABC 123",
    ]

def test_detect_repeated_lines(preprocessing):
    pages = [
        "Company Report\nPage one content",
        "Company Report\nPage two content",
        "Company Report\nPage three content",
        "Company Report\nPage four content",
    ]
    result = preprocessing.detect_repeated_lines(
        pages,
        top_n = 1,
        bottom_n = 1,
        threshold = 0.8
        )
    assert "Company Report" in result

def test_remove_repeated_lines(preprocessing):
    pages = [
        "Company Report\nContent page 1\nConfidential",
        "Company Report\nContent page 2\nConfidential",
        "Company Report\nContent page 3\nConfidential",
    ]
    repeated_lines = {
        "Company Report",
        "Confidential",
    }
    result = preprocessing.remove_repeated_lines(pages, repeated_lines)
    assert result == [
        "Content page 1",
        "Content page 2",
        "Content page 3",
    ]