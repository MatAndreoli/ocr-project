from pathlib import Path

from ocr_placas.comparacao import gerar_comparacao

RAIZ = Path(__file__).resolve().parents[1]


def test_gerar_comparacao_calcula_acerto_e_cer(tmp_path):
    pasta = tmp_path
    (pasta / "tesseract").mkdir()
    (pasta / "easyocr").mkdir()

    cabecalho = "imagem,texto_bruto,texto_normalizado,tempo_segundos\n"
    (pasta / "tesseract" / "resultados.csv").write_text(
        cabecalho + "1.JPG,IFA1T95,IFA1T95,0.1\n2.JPG,NEV1C08,NEV1C08,0.2\n"
    )
    (pasta / "easyocr" / "resultados.csv").write_text(
        cabecalho + "1.JPG,IFA1T95,IFA1T95,1.0\n2.JPG,NEV1C08,NEV1C09,1.2\n"
    )

    comparacao = gerar_comparacao(pasta, RAIZ / "ground_truth.csv")

    tesseract = {
        linha["imagem"]: linha
        for linha in comparacao
        if linha["metodo"] == "Tesseract"
    }
    easyocr = {
        linha["imagem"]: linha
        for linha in comparacao
        if linha["metodo"] == "EasyOCR"
    }

    assert tesseract["1.JPG"]["acerto_exato"] is True
    assert tesseract["2.JPG"]["cer"] == 0.0
    assert easyocr["2.JPG"]["acerto_exato"] is False
    assert easyocr["2.JPG"]["cer"] > 0.0
