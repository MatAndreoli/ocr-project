"""Localização e extração da placa veicular.

A detecção usa a faixa azul superior das placas Mercosul como âncora.
A faixa tem saturação alta e proporção horizontal bem definida, o que
permite localizá-la com poucas operações de cor e componentes conexos.
"""

from __future__ import annotations

import math
from typing import Any

import cv2


def localizar_placa(imagem: Any) -> tuple[int, int, int, int] | None:
    """Retorna (x, y, largura, altura) da placa ou None.

    A altura é estimada a partir da largura, usando a proporção aproximada
    de uma placa Mercosul (40 cm x 13 cm).
    """
    if imagem is None or imagem.size == 0:
        return None

    altura_imagem, largura_imagem = imagem.shape[:2]
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Azul saturado da faixa Mercosul. Saturação alta evita céu, vidros e
    # lataria clara, que costumam ter azul pouco saturado.
    mascara_azul = cv2.inRange(hsv, (95, 120, 40), (132, 255, 255))
    mascara_azul = cv2.morphologyEx(
        mascara_azul,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
    )

    _, _, estatisticas, _ = cv2.connectedComponentsWithStats(mascara_azul)
    melhor: tuple[float, int, int, int, int] | None = None

    for indice in range(1, len(estatisticas)):
        x, y, largura, altura_faixa, area = estatisticas[indice]

        # Filtros geométricos da faixa azul.
        if largura < largura_imagem * 0.08:
            continue
        if not 8 <= altura_faixa <= 90:
            continue
        if not 5 < largura / altura_faixa < 35:
            continue

        # Abaixo da faixa deve existir o corpo claro da placa.
        altura_placa = min(round(largura / 3.1), altura_imagem - y)
        if altura_placa < altura_faixa * 2:
            continue

        corpo_hsv = hsv[y + altura_faixa : y + altura_placa, x : x + largura]
        corpo_cinza = cinza[y + altura_faixa : y + altura_placa, x : x + largura]
        if corpo_hsv.size == 0:
            continue

        pixels_claros = (
            (corpo_hsv[:, :, 2] > 110) & (corpo_hsv[:, :, 1] < 90)
        ).mean()
        pixels_escuros = (corpo_cinza < 90).mean()
        if pixels_escuros < 0.02:
            continue

        # Corpo claro com caracteres escuros produz a maior pontuação.
        pontuacao = pixels_claros * (1 + pixels_escuros) * math.sqrt(area)
        if melhor is None or pontuacao > melhor[0]:
            melhor = (pontuacao, int(x), int(y), int(largura), int(altura_placa))

    if melhor is None:
        return None

    _, x, y, largura, altura = melhor
    return x, y, largura, altura


def extrair_placa(imagem: Any) -> Any:
    """Localiza a placa e retorna o recorte com uma pequena margem."""
    caixa = localizar_placa(imagem)
    if caixa is None:
        return None

    x, y, largura, altura = caixa
    margem = int(largura * 0.03)
    x1 = max(0, x - margem)
    y1 = max(0, y - margem)
    x2 = min(imagem.shape[1], x + largura + margem)
    y2 = min(imagem.shape[0], y + altura + margem)
    return imagem[y1:y2, x1:x2]


def desenhar_deteccao(
    imagem: Any, caixa: tuple[int, int, int, int]
) -> Any:
    """Desenha a caixa da placa sobre uma cópia da imagem."""
    saida = imagem.copy()
    x, y, largura, altura = caixa
    cv2.rectangle(saida, (x, y), (x + largura, y + altura), (0, 255, 0), 3)
    return saida
