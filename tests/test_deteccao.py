from pathlib import Path

import cv2

from ocr_placas.deteccao import extrair_placa, localizar_placa

RAIZ = Path(__file__).resolve().parents[1]
IMAGENS = sorted((RAIZ / "data").glob("*.JPG"), key=lambda p: int(p.stem))


def test_localiza_placa_na_primeira_imagem():
    imagem = cv2.imread(str(RAIZ / "data" / "1.JPG"))
    caixa = localizar_placa(imagem)

    assert caixa is not None
    x, y, largura, altura = caixa
    assert abs(x - 194) < 45
    assert abs(y - 234) < 45
    assert 300 < largura < 500
    assert 2.5 < largura / altura < 4.5


def test_extrai_placa_retorna_recorte_valido():
    imagem = cv2.imread(str(RAIZ / "data" / "1.JPG"))
    recorte = extrair_placa(imagem)

    assert recorte is not None
    assert recorte.ndim == 3
    assert recorte.shape[1] > 300
    assert recorte.shape[0] > 80


def test_localiza_placa_em_todas_as_imagens():
    falhas = [p.name for p in IMAGENS if localizar_placa(cv2.imread(str(p))) is None]
    assert falhas == []
