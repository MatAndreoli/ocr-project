from pathlib import Path

import cv2

from ocr_placas.deteccao import extrair_placa
from ocr_placas.preprocessamento import preparar_para_tesseract, recortar_caracteres

RAIZ = Path(__file__).resolve().parents[1]


def _placa(numero: int):
    imagem = cv2.imread(str(RAIZ / "data" / f"{numero}.JPG"))
    recorte = extrair_placa(imagem)
    assert recorte is not None
    return recorte


def test_preparar_para_tesseract_gera_imagem_binaria():
    binaria = preparar_para_tesseract(_placa(1))

    assert binaria.ndim == 2
    assert str(binaria.dtype) == "uint8"
    assert set(binaria.flatten().tolist()).issubset({0, 255})
    assert binaria.shape[1] > binaria.shape[0]


def test_recortar_caracteres_remove_faixa_azul_e_bordas():
    recorte = recortar_caracteres(_placa(1))

    assert recorte.ndim == 3
    assert recorte.shape[2] == 3
    assert recorte.shape[1] > recorte.shape[0] * 2
    # A faixa azul superior não deve dominar o recorte.
    topo = recorte[: recorte.shape[0] // 6]
    assert topo[:, :, 0].mean() - topo[:, :, 2].mean() < 50


def test_preparar_para_tesseract_mantem_resolucao_da_placa_grande():
    binaria = preparar_para_tesseract(_placa(3))

    assert binaria.shape[1] > 1140


def test_preprocessamento_funciona_em_placa_escura():
    binaria = preparar_para_tesseract(_placa(20))

    assert binaria.ndim == 2
    assert (binaria == 0).mean() > 0.02
