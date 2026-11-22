# OCR de Placas Veiculares

Pipeline de visão computacional para localizar, extrair e reconhecer placas
veiculares brasileiras (Mercosul), comparando **Tesseract OCR** e **EasyOCR**
sobre o mesmo conjunto de imagens.

## Resultados

| Método    | Placas exatas | Acurácia por caractere | Tempo médio/imagem |
| --------- | ------------: | ---------------------: | -----------------: |
| Tesseract |         10/20 |                  70,0% |             1,57 s |
| EasyOCR   |         13/20 |                  83,6% |             0,51 s |

O EasyOCR teve melhor acurácia geral. O Tesseract ficou mais lento por executar
um ensemble de recortes e configurações.

Relatório completo por imagem:
[`results/comparacao/relatorio.md`](results/comparacao/relatorio.md).

## Estrutura

```text
ocr-project/
├── data/                  # imagens de entrada
├── ground_truth.csv       # placas esperadas
├── ocr_placas/
│   ├── deteccao.py        # localiza e recorta a placa
│   ├── preprocessamento.py# prepara as imagens para OCR
│   ├── ocr.py             # Tesseract, EasyOCR e normalização
│   ├── pipeline.py        # executa e grava os resultados
│   └── comparacao.py      # métricas e relatório
├── run_tesseract.py
├── run_easyocr.py
├── results/
└── tests/
```

## Requisitos

- Python 3.11 ou 3.12
- Tesseract instalado no sistema

Arch Linux:

```bash
sudo pacman -S tesseract tesseract-data-eng
```

Ubuntu/Debian:

```bash
sudo apt install tesseract-ocr tesseract-ocr-eng
```

## Instalação

### Com uv

```bash
uv python pin 3.11
uv sync --extra dev
```

### Com pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se não tiver Python 3.11 no sistema, o uv pode criar o ambiente:

```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Como rodar

### Tesseract

```bash
uv run python run_tesseract.py
```

### EasyOCR

```bash
uv run python run_easyocr.py
```

### Comparação

```bash
uv run python -m ocr_placas.comparacao
```

Com pip, troque `uv run python` por `python`.

Na primeira execução, o EasyOCR baixa os modelos para `.cache/easyocr/`.

## Como o pipeline funciona

1. **Detecção:** a faixa azul da placa Mercosul é localizada com OpenCV/HSV e
   usada para estimar o retângulo da placa.
2. **Extração:** a placa é recortada e a região dos caracteres é isolada.
3. **Pré-processamento:** são geradas variantes com CLAHE, Otsu, limiar fixo e
   diferentes recortes.
4. **OCR:**
   - Tesseract: ensemble das variantes com `psm 7` e `psm 8`;
   - EasyOCR: reconhecimento por caractere com whitelist posicional
     (letras/números) e fallback para a imagem inteira.
5. **Pós-processamento:** limpeza e normalização para o formato Mercosul
   `LLLNLNN`.
6. **Saída:** um `.txt` por imagem e um CSV consolidado por método.

## Saídas

- `results/tesseract/<imagem>.txt`
- `results/easyocr/<imagem>.txt`
- `results/tesseract/resultados.csv`
- `results/easyocr/resultados.csv`
- `results/comparacao/comparacao.csv`
- `results/comparacao/relatorio.md`
- `results/placas/` — recortes, imagens anotadas e processadas

## Testes

```bash
uv run pytest
```

## Observações

- O dataset tem 20 imagens de placas Mercosul.
- A detecção depende da faixa azul; placas sem faixa não são suportadas.
- Placas muito escuras, pequenas ou com reflexo ainda podem falhar.
- O texto bruto de cada OCR é preservado nos CSVs.
