"""Orquestração do pipeline de segmentação + OCR."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import cv2

from .deteccao import desenhar_deteccao, extrair_placa, localizar_placa
from .ocr import criar_leitor_easyocr, ler_com_easyocr, ler_com_tesseract
from .preprocessamento import preparar_para_tesseract, recortar_caracteres

CAMPOS_CSV = [
    "imagem",
    "texto_bruto",
    "texto_normalizado",
    "tempo_segundos",
]


def executar_pipeline(
    backend: str,
    imagens: list[Path],
    pasta_saida: Path | str = "results",
) -> list[dict[str, Any]]:
    """Processa as imagens e grava recortes, textos e CSV.

    backend: "tesseract" ou "easyocr".
    Retorna uma lista de dicionários, um por imagem processada.
    """
    if backend not in {"tesseract", "easyocr"}:
        raise ValueError("backend deve ser 'tesseract' ou 'easyocr'")

    pasta_saida = Path(pasta_saida)
    pasta_backend = pasta_saida / backend
    pasta_placas = pasta_saida / "placas"
    pasta_backend.mkdir(parents=True, exist_ok=True)
    pasta_placas.mkdir(parents=True, exist_ok=True)

    leitor_easyocr = criar_leitor_easyocr() if backend == "easyocr" else None
    registros: list[dict[str, Any]] = []

    for caminho in imagens:
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            print(f"[aviso] não foi possível ler {caminho}")
            continue

        nome = Path(caminho).stem
        caixa = localizar_placa(imagem)
        texto_bruto = ""
        texto_normalizado = ""
        tempo = 0.0

        if caixa is None:
            print(f"[aviso] placa não localizada em {caminho.name}")
        else:
            placa = extrair_placa(imagem)
            if placa is not None:
                inicio = time.perf_counter()
                if backend == "tesseract":
                    texto_bruto, texto_normalizado = ler_com_tesseract(placa)
                    processada = preparar_para_tesseract(placa)
                else:
                    texto_bruto, texto_normalizado = ler_com_easyocr(
                        placa, leitor_easyocr
                    )
                    processada = recortar_caracteres(placa)
                tempo = time.perf_counter() - inicio

                cv2.imwrite(str(pasta_placas / f"{nome}.png"), placa)
                cv2.imwrite(str(pasta_placas / f"{nome}_processada.png"), processada)
                cv2.imwrite(
                    str(pasta_placas / f"{nome}_anotada.jpg"),
                    desenhar_deteccao(imagem, caixa),
                )

        (pasta_backend / f"{nome}.txt").write_text(
            texto_normalizado + "\n", encoding="utf-8"
        )
        registros.append(
            {
                "imagem": Path(caminho).name,
                "texto_bruto": texto_bruto,
                "texto_normalizado": texto_normalizado,
                "tempo_segundos": round(tempo, 4),
            }
        )
        print(f"{Path(caminho).name}: {texto_normalizado or '(vazio)'}")

    with (pasta_backend / "resultados.csv").open(
        "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=CAMPOS_CSV)
        escritor.writeheader()
        escritor.writerows(registros)

    return registros
