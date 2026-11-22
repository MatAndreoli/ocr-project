"""Preparação da placa para os motores de OCR.

O pré-processamento gera algumas variantes de recorte e binarização. Isso é
necessário porque placas escuras, com reflexo ou com fundo claro reagem de
forma diferente a CLAHE, Otsu e limiar fixo.
"""

from __future__ import annotations

from typing import Any

import cv2

# Variantes testadas: (nome, topo ajustado, CLAHE, limiar)
# "topo ajustado" usa o fim da faixa azul para não cortar o topo dos caracteres.
_VARIANTES = [
    ("fixo-clahe-otsu", False, True, "otsu"),
    ("ajustado-clahe-otsu", True, True, "otsu"),
    ("ajustado-otsu", True, False, "otsu"),
    ("fixo-otsu", False, False, "otsu"),
    ("fixo-clahe-90", False, True, 90),
    ("ajustado-clahe-90", True, True, 90),
    ("ajustado-90", True, False, 90),
]


def _fim_faixa_azul(placa: Any) -> int | None:
    """Retorna a última linha da primeira faixa azul encontrada."""
    altura, largura = placa.shape[:2]
    hsv = cv2.cvtColor(placa, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, (95, 120, 40), (132, 255, 255))
    linhas_azuis = (mascara > 0).sum(axis=1)
    ativas = linhas_azuis > largura * 0.15

    inicio = None
    for indice, ativa in enumerate(ativas):
        if ativa and inicio is None:
            inicio = indice
        if inicio is not None and (not ativa or indice == len(ativas) - 1):
            fim = indice if not ativa else indice + 1
            if fim - inicio >= 4 and inicio < altura * 0.4:
                return fim
            inicio = None
    return None


def _corpo_da_placa(
    placa: Any, usar_clahe: bool, topo_ajustado: bool = False
) -> tuple[Any, Any]:
    """Remove a faixa azul, bordas e retorna corpo colorido + cinza."""
    altura, largura = placa.shape[:2]
    topo = int(altura * 0.24)

    if topo_ajustado:
        fim_azul = _fim_faixa_azul(placa)
        if fim_azul is not None:
            # Só sobe o topo; nunca desce além do recorte padrão.
            topo = min(topo, fim_azul + 1)

    corpo = placa[topo : int(altura * 0.82), int(largura * 0.06) : int(largura * 0.99)]

    cinza = cv2.cvtColor(corpo, cv2.COLOR_BGR2GRAY)
    if usar_clahe:
        # Realça o contraste em placas escuras ou com reflexo.
        cinza = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cinza)
    return corpo, cinza


def _binaria(cinza: Any, limiar: str | int) -> Any:
    """Binariza a imagem ampliada 3x."""
    ampliada = cv2.resize(cinza, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    if limiar == "otsu":
        return cv2.threshold(
            ampliada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
    return cv2.threshold(ampliada, int(limiar), 255, cv2.THRESH_BINARY)[1]


def _caixas_caracteres(binaria: Any) -> list[tuple[int, int, int, int]]:
    """Encontra componentes altos e estreitos com aparência de caractere."""
    altura, largura = binaria.shape
    contornos, _ = cv2.findContours(
        255 - binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    caixas = []
    for contorno in contornos:
        x, y, largura_caixa, altura_caixa = cv2.boundingRect(contorno)
        if not 0.30 * altura < altura_caixa:
            continue
        if not 0.008 * largura < largura_caixa < 0.20 * largura:
            continue
        if y <= 0.02 * altura:
            continue
        if not 0.10 < largura_caixa / altura_caixa < 1.40:
            continue
        caixas.append((x, y, largura_caixa, altura_caixa))
    return caixas


def _faixa_caracteres(
    caixas: list[tuple[int, int, int, int]], largura: int, altura: int
) -> tuple[int, int, int, int] | None:
    """Une as caixas dos caracteres em uma faixa horizontal."""
    if len(caixas) < 4:
        return None

    x1 = min(x for x, _, _, _ in caixas)
    y1 = min(y for _, y, _, _ in caixas)
    x2 = max(x + largura_caixa for x, _, largura_caixa, _ in caixas)
    y2 = max(y + altura_caixa for _, y, _, altura_caixa in caixas)

    margem_y = int((y2 - y1) * 0.15)
    return (
        max(0, x1),
        max(0, y1 - margem_y),
        min(largura, x2),
        min(altura, y2 + margem_y),
    )


def _variantes(placa: Any):
    """Gera as variantes de corpo, binarização e caixas de caracteres."""
    for nome, topo_ajustado, usar_clahe, limiar in _VARIANTES:
        corpo, cinza = _corpo_da_placa(placa, usar_clahe, topo_ajustado)
        binaria = _binaria(cinza, limiar)
        caixas = _caixas_caracteres(binaria)
        yield {
            "nome": nome,
            "corpo": corpo,
            "binaria": binaria,
            "caixas": caixas,
        }


def preparar_para_tesseract(placa: Any) -> Any:
    """Retorna a variante padrão binária, com borda branca."""
    corpo, cinza = _corpo_da_placa(placa, usar_clahe=True)
    binaria = _binaria(cinza, "otsu")
    caixas = _caixas_caracteres(binaria)

    faixa = _faixa_caracteres(caixas, binaria.shape[1], binaria.shape[0])
    if faixa is not None:
        x1, y1, x2, y2 = faixa
        binaria = binaria[y1:y2, x1:x2]

    # Tesseract trabalha melhor com margem branca ao redor do texto.
    return cv2.copyMakeBorder(
        binaria, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
    )


def preparar_variantes_tesseract(placa: Any) -> list[tuple[str, Any]]:
    """Retorna as imagens binárias das variantes para o Tesseract."""
    imagens = []
    for variante in _variantes(placa):
        binaria = variante["binaria"]
        faixa = _faixa_caracteres(
            variante["caixas"], binaria.shape[1], binaria.shape[0]
        )
        if faixa is not None:
            x1, y1, x2, y2 = faixa
            binaria = binaria[y1:y2, x1:x2]

        imagens.append(
            (
                variante["nome"],
                cv2.copyMakeBorder(
                    binaria, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255
                ),
            )
        )
    return imagens


def segmentar_caracteres(placa: Any) -> list[dict[str, Any]]:
    """Retorna variantes com pelo menos 3 caixas de caracteres."""
    return [
        variante
        for variante in _variantes(placa)
        if len(variante["caixas"]) >= 3
    ]


def recortar_caracteres(placa: Any, usar_clahe: bool = True) -> Any:
    """Retorna recorte colorido apenas com a região dos caracteres."""
    corpo, cinza = _corpo_da_placa(placa, usar_clahe=usar_clahe)
    binaria = _binaria(cinza, "otsu")
    caixas = _caixas_caracteres(binaria)
    faixa = _faixa_caracteres(caixas, binaria.shape[1], binaria.shape[0])

    if faixa is None:
        return corpo

    # A faixa foi calculada na imagem ampliada 3x.
    x1, y1, x2, y2 = (valor // 3 for valor in faixa)
    margem_x = int((x2 - x1) * 0.05)
    margem_y = int((y2 - y1) * 0.25)
    return corpo[
        max(0, y1 - margem_y) : min(corpo.shape[0], y2 + margem_y),
        max(0, x1 - margem_x) : min(corpo.shape[1], x2 + margem_x),
    ]
