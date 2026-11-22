"""Executa o pipeline completo com Tesseract OCR."""

from pathlib import Path

from ocr_placas.pipeline import executar_pipeline

RAIZ = Path(__file__).resolve().parent


def main() -> None:
    imagens = sorted((RAIZ / "data").glob("*.JPG"), key=lambda p: int(p.stem))
    resultados = executar_pipeline("tesseract", imagens, RAIZ / "results")
    print(f"\n{len(resultados)} imagens processadas com Tesseract.")


if __name__ == "__main__":
    main()
