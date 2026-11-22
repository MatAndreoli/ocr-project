import pytest

from ocr_placas.comparacao import acuracia_caracteres, cer, distancia_edicao


def test_distancia_edicao_identica():
    assert distancia_edicao("IFA1T95", "IFA1T95") == 0


def test_distancia_edicao_um_caractere():
    assert distancia_edicao("IFA1T95", "IFA1T05") == 1


def test_acuracia_caracteres_identica():
    assert acuracia_caracteres("IFA1T95", "IFA1T95") == 1.0


def test_acuracia_caracteres_com_um_erro():
    assert acuracia_caracteres("IFA1T95", "IFA1T05") == pytest.approx(6 / 7)


def test_cer_com_um_erro():
    assert cer("IFA1T95", "IFA1T05") == pytest.approx(1 / 7)
