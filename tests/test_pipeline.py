from pathlib import Path

from ocr_placas.pipeline import executar_pipeline

RAIZ = Path(__file__).resolve().parents[1]


def test_pipeline_tesseract_gera_txt_e_csv(tmp_path):
    resultado = executar_pipeline(
        backend="tesseract",
        imagens=[RAIZ / "data" / "1.JPG"],
        pasta_saida=tmp_path,
    )

    assert (tmp_path / "tesseract" / "1.txt").read_text().strip() == "IFA1T95"
    assert (tmp_path / "tesseract" / "resultados.csv").exists()
    assert resultado[0]["texto_normalizado"] == "IFA1T95"
