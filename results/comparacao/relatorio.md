# Comparação Tesseract × EasyOCR

## Resumo

| Método | Acertos exatos | Acurácia exata | Acurácia por caractere | CER médio | Tempo médio (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tesseract | 10/20 | 50.0% | 70.0% | 30.0% | 1.573 |
| EasyOCR | 13/20 | 65.0% | 83.6% | 16.4% | 0.508 |

## Resultados por imagem

| Imagem | Placa esperada | Tesseract | EasyOCR |
| --- | --- | --- | --- |
| 1.JPG | IFA1T95 | IFA1T95 | IFA1T95 |
| 2.JPG | NEV1C08 | NEV1C68 | NEV1C08 |
| 3.JPG | IYZ2Z18 | IYZ2Z18 | IVZ2Z18 |
| 4.JPG | QOW9N47 | QOW9N47 | QOW9N47 |
| 5.JPG | USQ0T32 | 11942711 | USQL4Z |
| 6.JPG | TYL3X75 | JMQ0OM1 | TVL |
| 7.JPG | GXQ2J95 | 775A4L22 | GXQ2 |
| 8.JPG | XHY6U29 | XHY4U29 | XHY6U29 |
| 9.JPG | AOA4J16 | AOA4J16 | AOA4J16 |
| 10.JPG | PDU7G22 | PDU7G22 | PDU7G22 |
| 11.JPG | MNL3U98 | MNL3U98 | MNL3U98 |
| 12.JPG | JET7G88 | JET7G88 | JET7G88 |
| 13.JPG | ABP8Q89 | IABPB89 | ABP8O82 |
| 14.JPG | PHI8H00 | PHI8H66 | PHI8H00 |
| 15.JPG | THY7E19 | THY2E19 | THV7E19 |
| 16.JPG | UHK5O55 | UHK5O55 | UHK5O55 |
| 17.JPG | NZQ3M77 | NZQ3M77 | NZQ3M77 |
| 18.JPG | XDA5E50 | LXBASESD | XDA5E50 |
| 19.JPG | ESI4I93 | ESI4I93 | ESI4I93 |
| 20.JPG | TGG1D44 | AHI4L54 | R |

## Observações

- O pós-processamento converte o texto para maiúsculas, remove símbolos e
  corrige confusões comuns respeitando o formato Mercosul `LLLNLNN`.
- O texto bruto de cada OCR é preservado em `results/<método>/resultados.csv`.
- CER é calculado como `1 - acurácia por caractere`.
