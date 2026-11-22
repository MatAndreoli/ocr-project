import pytest

from ocr_placas.ocr import limpar_texto, normalizar_placa, pontuar_placa


def test_limpar_texto_remove_espacos_e_simbolos():
    assert limpar_texto(" ifa-1t95\n") == "IFA1T95"


def test_limpar_texto_pode_retornar_vazio():
    assert limpar_texto("--- ...") == ""


def test_normalizar_placa_corrige_confusoes_posicionais():
    assert normalizar_placa("1FAIT9S") == "IFA1T95"


def test_normalizar_placa_nao_altera_texto_fora_do_formato():
    assert normalizar_placa("ABC") == "ABC"


def test_pontuar_placa_prefere_formato_valido():
    assert pontuar_placa("IFA1T95") > pontuar_placa("IFA1T9")
