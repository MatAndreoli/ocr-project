"""Pós-processamento de texto e backends de OCR."""

from __future__ import annotations

import re
import statistics
from typing import Any

import cv2
import pytesseract

from .preprocessamento import (
    preparar_para_tesseract,
    preparar_variantes_tesseract,
    recortar_caracteres,
    segmentar_caracteres,
)

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LETRAS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITOS = "0123456789"

# Conversões usadas apenas quando o texto já tem 7 caracteres.
# O padrão Mercosul é LLLNLNN (letra, letra, letra, número, letra, número, número).
_DIGITO_PARA_LETRA = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}
_LETRA_PARA_DIGITO = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "A": "4",
    "G": "6",
    "T": "7",
    "J": "7",
}


def limpar_texto(texto: str) -> str:
    """Maiúsculas e somente letras/números."""
    return re.sub(r"[^A-Z0-9]", "", texto.upper())


def normalizar_placa(texto: str) -> str:
    """Corrige confusões comuns respeitando o formato LLLNLNN."""
    limpo = limpar_texto(texto)
    if len(limpo) != 7:
        return limpo

    saida = []
    for indice, caractere in enumerate(limpo):
        if indice in (0, 1, 2, 4):
            saida.append(_DIGITO_PARA_LETRA.get(caractere, caractere))
        else:
            saida.append(_LETRA_PARA_DIGITO.get(caractere, caractere))
    return "".join(saida)


def pontuar_placa(texto: str) -> int:
    """Pontua o quão próximo o texto está de uma placa Mercosul válida."""
    normalizado = normalizar_placa(texto)
    pontuacao = 20 if len(normalizado) == 7 else -abs(len(normalizado) - 7) * 2

    if len(normalizado) == 7:
        if normalizado[0:3].isalpha():
            pontuacao += 3
        if normalizado[3].isdigit():
            pontuacao += 3
        if normalizado[4].isalpha():
            pontuacao += 3
        if normalizado[5:7].isdigit():
            pontuacao += 3
    return pontuacao


def _ocr_linha_tesseract(imagem: Any) -> tuple[str, float]:
    """Executa Tesseract nos modos 7 e 8 e retorna o melhor texto/confiança."""
    melhor = ("", 0.0)

    for psm in (7, 8):
        config = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
        try:
            dados = pytesseract.image_to_data(
                imagem, config=config, output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractError:
            continue

        palavras = [
            (texto.strip(), float(confianca))
            for texto, confianca in zip(dados["text"], dados["conf"])
            if texto.strip() and float(confianca) >= 0
        ]
        if not palavras:
            continue

        texto = limpar_texto("".join(palavra for palavra, _ in palavras))
        confianca = sum(conf for _, conf in palavras) / len(palavras)
        if (pontuar_placa(texto), confianca) > (
            pontuar_placa(melhor[0]),
            melhor[1],
        ):
            melhor = (texto, confianca)

    return melhor


def ler_com_tesseract(placa: Any) -> tuple[str, str]:
    """Executa Tesseract e retorna (texto bruto, texto normalizado)."""
    candidatos = []

    # Ensemble de recortes/binarizações: escolhe o resultado com melhor
    # formato de placa e, em caso de empate, maior confiança.
    for _, imagem in preparar_variantes_tesseract(placa):
        candidatos.append(_ocr_linha_tesseract(imagem))

    # Tesseract pode falhar em imagens muito largas. Se nenhum candidato tiver
    # o tamanho esperado, tenta uma versão reduzida da imagem padrão.
    if not any(len(normalizar_placa(texto)) == 7 for texto, _ in candidatos):
        imagem = preparar_para_tesseract(placa)
        if imagem.shape[1] > 1140:
            escala = 1100 / imagem.shape[1]
            reduzida = cv2.resize(
                imagem,
                None,
                fx=escala,
                fy=escala,
                interpolation=cv2.INTER_AREA,
            )
            candidatos.append(_ocr_linha_tesseract(reduzida))

    melhor = max(candidatos, key=lambda item: (pontuar_placa(item[0]), item[1]))
    return melhor[0], normalizar_placa(melhor[0])


def criar_leitor_easyocr() -> Any:
    """Cria o leitor EasyOCR uma única vez, em CPU."""
    import easyocr

    return easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=".cache/easyocr",
        download_enabled=True,
        verbose=False,
    )


def _selecionar_sete(caixas: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Mantém no máximo 7 caixas, removendo as mais fora do padrão."""
    caixas = sorted(caixas)
    if len(caixas) <= 7:
        return caixas

    largura_mediana = statistics.median(caixa[2] for caixa in caixas)
    altura_mediana = statistics.median(caixa[3] for caixa in caixas)
    caixas = list(caixas)

    while len(caixas) > 7:
        indice = max(
            range(len(caixas)),
            key=lambda i: (
                abs(caixas[i][2] - largura_mediana) / largura_mediana
                + abs(caixas[i][3] - altura_mediana) / altura_mediana
            ),
        )
        caixas.pop(indice)

    return sorted(caixas)


def _ler_caracteres_easyocr(
    binaria: Any,
    caixas: list[tuple[int, int, int, int]],
    leitor: Any,
) -> tuple[str, float]:
    """Reconhece caracteres individuais com whitelist posicional."""
    altura, largura = binaria.shape
    caracteres = []
    confiancas = []

    for indice, (x, y, largura_caixa, altura_caixa) in enumerate(
        _selecionar_sete(caixas)
    ):
        recorte = binaria[
            max(0, y - 4) : min(altura, y + altura_caixa + 4),
            max(0, x - 4) : min(largura, x + largura_caixa + 4),
        ]
        recorte = cv2.resize(
            recorte, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
        )
        recorte = cv2.copyMakeBorder(
            recorte, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255
        )
        recorte = cv2.cvtColor(recorte, cv2.COLOR_GRAY2BGR)

        whitelist = LETRAS if indice in (0, 1, 2, 4) else DIGITOS
        resultados = leitor.recognize(
            recorte,
            allowlist=whitelist,
            detail=1,
            decoder="beamsearch",
            beamWidth=5,
        )

        if resultados:
            texto, confianca = max(
                ((resultado[1], float(resultado[2])) for resultado in resultados),
                key=lambda item: item[1],
            )
            caracteres.append(limpar_texto(texto)[:1])
            confiancas.append(confianca)
        else:
            caracteres.append("")
            confiancas.append(0.0)

    confianca_media = sum(confiancas) / len(confiancas) if confiancas else 0.0
    return "".join(caracteres), confianca_media


def _ler_imagem_inteira_easyocr(placa: Any, leitor: Any) -> tuple[str, float]:
    """Fallback: executa EasyOCR sobre o recorte inteiro dos caracteres."""
    imagem = recortar_caracteres(placa, usar_clahe=True)
    borda_y = int(imagem.shape[0] * 0.20)
    borda_x = int(imagem.shape[1] * 0.05)
    imagem = cv2.copyMakeBorder(
        imagem,
        borda_y,
        borda_y,
        borda_x,
        borda_x,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    resultados = leitor.readtext(
        imagem,
        detail=1,
        paragraph=False,
        allowlist=WHITELIST,
        mag_ratio=2,
    )
    if not resultados:
        return "", 0.0

    textos = [limpar_texto(resultado[1]) for resultado in resultados]
    textos = [texto for texto in textos if texto]
    if not textos:
        return "", 0.0

    # Também considera a concatenação dos segmentos, da esquerda para a direita.
    ordenados = sorted(resultados, key=lambda resultado: resultado[0][0][0])
    textos.append(limpar_texto(" ".join(resultado[1] for resultado in ordenados)))

    melhor = max(textos, key=pontuar_placa)
    confianca = sum(float(resultado[2]) for resultado in resultados) / len(resultados)
    return melhor, confianca


def ler_com_easyocr(placa: Any, leitor: Any) -> tuple[str, str]:
    """Executa EasyOCR e retorna (texto bruto, texto normalizado).

    A estratégia principal segmenta os caracteres com OpenCV e usa o
    reconhecedor do EasyOCR com whitelist por posição (letra/número).
    Se a segmentação falhar, usa o detector CRAFT na imagem inteira.
    """
    candidatos = []

    for variante in segmentar_caracteres(placa):
        texto, confianca = _ler_caracteres_easyocr(
            variante["binaria"], variante["caixas"], leitor
        )
        if texto:
            candidatos.append((texto, confianca))

    melhor = (
        max(candidatos, key=lambda item: (pontuar_placa(item[0]), item[1]))
        if candidatos
        else None
    )

    # Fallback para placas em que a segmentação retornou poucos caracteres.
    if melhor is None or len(normalizar_placa(melhor[0])) <= 3:
        texto, confianca = _ler_imagem_inteira_easyocr(placa, leitor)
        if texto:
            candidatos.append((texto, confianca))
            melhor = max(
                candidatos, key=lambda item: (pontuar_placa(item[0]), item[1])
            )

    if melhor is None:
        return "", ""

    return melhor[0], normalizar_placa(melhor[0])
