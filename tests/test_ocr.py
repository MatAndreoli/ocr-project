from pathlib import Path

import cv2
import pytest

from ocr_placas.deteccao import extrair_placa
from ocr_placas.ocr import criar_leitor_easyocr, ler_com_easyocr, ler_com_tesseract

RAIZ = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def leitor_easyocr():
    return criar_leitor_easyocr()


def _placa(numero: int):
    imagem = cv2.imread(str(RAIZ / "data" / f"{numero}.JPG"))
    recorte = extrair_placa(imagem)
    assert recorte is not None
    return recorte


def test_tesseract_reconhece_primeira_placa():
    bruto, normalizado = ler_com_tesseract(_placa(1))

    assert bruto
    assert normalizado == "IFA1T95"


def test_tesseract_reconhece_placa_com_ruido():
    bruto, normalizado = ler_com_tesseract(_placa(10))

    assert bruto
    assert normalizado == "PDU7G22"


def test_tesseract_processa_placa_grande_sem_retornar_vazio():
    bruto, normalizado = ler_com_tesseract(_placa(2))

    assert bruto or normalizado


def test_easyocr_reconhece_primeira_placa(leitor_easyocr):
    bruto, normalizado = ler_com_easyocr(_placa(1), leitor_easyocr)

    assert bruto
    assert normalizado == "IFA1T95"


def test_easyocr_reconhece_segunda_placa(leitor_easyocr):
    bruto, normalizado = ler_com_easyocr(_placa(2), leitor_easyocr)

    assert bruto
    assert normalizado == "NEV1C08"

