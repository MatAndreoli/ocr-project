"""Métricas de comparação entre ground truth e textos reconhecidos."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any


def distancia_edicao(esperado: str, obtido: str) -> int:
    """Distância de Levenshtein entre duas strings."""
    linhas = len(esperado)
    colunas = len(obtido)
    anterior = list(range(colunas + 1))

    for i in range(1, linhas + 1):
        atual = [i] + [0] * colunas
        for j in range(1, colunas + 1):
            custo = 0 if esperado[i - 1] == obtido[j - 1] else 1
            atual[j] = min(
                anterior[j] + 1,
                atual[j - 1] + 1,
                anterior[j - 1] + custo,
            )
        anterior = atual

    return anterior[colunas]


def acuracia_caracteres(esperado: str, obtido: str) -> float:
    """Proporção de caracteres corretos, baseada na distância de edição."""
    return 1.0 - distancia_edicao(esperado, obtido) / max(len(esperado), 1)


def cer(esperado: str, obtido: str) -> float:
    """Character Error Rate (1 - acurácia por caractere)."""
    return 1.0 - acuracia_caracteres(esperado, obtido)


def _ler_csv(caminho: Path | str) -> list[dict[str, str]]:
    with Path(caminho).open(newline="", encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def gerar_comparacao(
    pasta_resultados: Path | str, caminho_ground_truth: Path | str
) -> list[dict[str, Any]]:
    """Combina ground truth, textos reconhecidos e métricas dos dois OCRs."""
    pasta_resultados = Path(pasta_resultados)
    ground_truth = _ler_csv(caminho_ground_truth)

    comparacao: list[dict[str, Any]] = []
    metodos = [
        ("tesseract", "Tesseract"),
        ("easyocr", "EasyOCR"),
    ]

    for pasta, rotulo in metodos:
        caminho = pasta_resultados / pasta / "resultados.csv"
        if not caminho.exists():
            continue

        dados = {linha["imagem"]: linha for linha in _ler_csv(caminho)}

        for referencia in ground_truth:
            imagem = referencia["imagem"]
            placa = referencia["placa"]
            resultado = dados.get(imagem, {})
            texto_bruto = resultado.get("texto_bruto", "")
            texto_normalizado = resultado.get("texto_normalizado", "")
            tempo = float(resultado.get("tempo_segundos") or 0.0)

            comparacao.append(
                {
                    "imagem": imagem,
                    "placa": placa,
                    "texto_bruto": texto_bruto,
                    "texto_normalizado": texto_normalizado,
                    "tempo_segundos": tempo,
                    "metodo": rotulo,
                    "acerto_exato": placa == texto_normalizado,
                    "acuracia_caracteres": acuracia_caracteres(
                        placa, texto_normalizado
                    ),
                    "cer": cer(placa, texto_normalizado),
                }
            )

    return comparacao


def resumir_comparacao(comparacao: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resume acurácia e tempo por método de OCR."""
    por_metodo: dict[str, list[dict[str, Any]]] = {}
    for linha in comparacao:
        por_metodo.setdefault(linha["metodo"], []).append(linha)

    resumo = []
    for metodo, linhas in por_metodo.items():
        total = len(linhas)
        acertos = sum(1 for linha in linhas if linha["acerto_exato"])
        resumo.append(
            {
                "metodo": metodo,
                "acertos": acertos,
                "total": total,
                "acuracia_exata": acertos / total if total else 0.0,
                "acuracia_caracteres_media": mean(
                    linha["acuracia_caracteres"] for linha in linhas
                ),
                "cer_medio": mean(linha["cer"] for linha in linhas),
                "tempo_medio_segundos": mean(
                    linha["tempo_segundos"] for linha in linhas
                ),
            }
        )
    return resumo


def salvar_relatorio(
    comparacao: list[dict[str, Any]],
    resumo: list[dict[str, Any]],
    caminho: Path | str,
) -> None:
    """Gera relatório Markdown com resumo e resultados por imagem."""
    caminho = Path(caminho)
    linhas = [
        "# Comparação Tesseract × EasyOCR",
        "",
        "## Resumo",
        "",
        "| Método | Acertos exatos | Acurácia exata | Acurácia por caractere | CER médio | Tempo médio (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for linha in resumo:
        linhas.append(
            "| {metodo} | {acertos:.0f}/{total} | {acuracia_exata:.1%} | "
            "{acuracia_caracteres_media:.1%} | {cer_medio:.1%} | {tempo_medio_segundos:.3f} |".format(
                **linha
            )
        )

    linhas.extend(
        [
            "",
            "## Resultados por imagem",
            "",
            "| Imagem | Placa esperada | Tesseract | EasyOCR |",
            "| --- | --- | --- | --- |",
        ]
    )

    por_imagem: dict[tuple[str, str], dict[str, str]] = {}
    for linha in comparacao:
        chave = (linha["imagem"], linha["placa"])
        por_imagem.setdefault(chave, {})[linha["metodo"]] = linha["texto_normalizado"]

    for (imagem, placa), textos in sorted(
        por_imagem.items(), key=lambda item: int(item[0][0].split(".")[0])
    ):
        tesseract = textos.get("Tesseract", "")
        easyocr = textos.get("EasyOCR", "")
        linhas.append(
            f"| {imagem} | {placa} | {tesseract or '—'} | {easyocr or '—'} |"
        )

    linhas.extend(
        [
            "",
            "## Observações",
            "",
            "- O pós-processamento converte o texto para maiúsculas, remove símbolos e",
            "  corrige confusões comuns respeitando o formato Mercosul `LLLNLNN`.",
            "- O texto bruto de cada OCR é preservado em `results/<método>/resultados.csv`.",
            "- CER é calculado como `1 - acurácia por caractere`.",
        ]
    )

    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _salvar_csv(comparacao: list[dict[str, Any]], caminho: Path | str) -> None:
    if not comparacao:
        return

    with Path(caminho).open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(comparacao[0]))
        escritor.writeheader()
        escritor.writerows(comparacao)


def main() -> None:
    raiz = Path.cwd()
    pasta_resultados = raiz / "results"
    pasta_comparacao = pasta_resultados / "comparacao"
    pasta_comparacao.mkdir(parents=True, exist_ok=True)

    comparacao = gerar_comparacao(pasta_resultados, raiz / "ground_truth.csv")
    resumo = resumir_comparacao(comparacao)

    _salvar_csv(comparacao, pasta_comparacao / "comparacao.csv")
    salvar_relatorio(comparacao, resumo, pasta_comparacao / "relatorio.md")

    for linha in resumo:
        print(
            f"{linha['metodo']}: {linha['acertos']}/{linha['total']} exatos | "
            f"acurácia por caractere {linha['acuracia_caracteres_media']:.1%} | "
            f"tempo médio {linha['tempo_medio_segundos']:.3f}s"
        )
    print(f"\nArquivos gerados em {pasta_comparacao}")


if __name__ == "__main__":
    main()
