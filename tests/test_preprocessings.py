from preprocessing.utils.utils import remove_links


def test_link_preprocessing():
    test_text = "Привет брат расскажи что это https://search.wb.ru/exactmatch/ru/common/v4/search? и что с этим делать"
    print(test_text)
    assert remove_links(test_text) == "Привет брат расскажи что это  и что с этим делать"
