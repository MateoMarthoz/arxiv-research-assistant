from backend.data_utils import *
from backend.extract_filter import classify_and_extract
from pprint import pprint

# Test that classify_and_extract correctly parses a natural language query into filter fields
def test_classification():
    standard_input = "Find me 3 papers on deep learning by ilya sutskever published before 2018 that doesn't include reinforcement learning"
    classification = classify_and_extract(standard_input)

    expected_filters = {
        "query": "Deep Learning",
        "author": "Ilya Sutskever",
        "title": None,
        "category": None,
        "abstract": None,
        "journal_ref": None,
        "doi": None,
        "exclude_words": "Reinforcement Learning",
        "start_date": None,
        "end_date": "20171231",
        "max_results": '3',
        "sort_by": "Relevance"
    }

    for key, value in expected_filters.items():
        actual = classification['filters'].get(key)
        expected = value

        if actual is None or expected is None:
            assert actual == expected
        else:
            assert actual.lower() == expected.lower()


# Test ability to extract filters from a minimal input
def test_small_classification():
    small_input = 'Find me the 10 latest papers by ilya sutskever'
    classification = classify_and_extract(small_input)

    expected_filters = {
        "query": None,
        "author": "Ilya Sutskever",
        "title": None,
        "category": None,
        "abstract": None,
        "journal_ref": None,
        "doi": None,
        "exclude_words": None,
        "start_date": None,
        "end_date": None,
        "max_results": '10',
        "sort_by": "newest"
    }

    for key, value in expected_filters.items():
        actual = classification['filters'].get(key)
        expected = value

        if actual is None or expected is None:
            assert actual == expected
        else:
            assert actual.lower() == expected.lower()    

# Test ability to extract filters from a complex input
def test_full_classification():
    full_input = 'Find me a paper on deep learning by ilya sutskever from 2010-2017 with "attention" in the title and NLP in the abstract.' \
    'It must be in the computation and language category and have been published on NeurIPS with doi: 10.1234/abcde.' \
    'I dont want it to be on computer vision though.'
    classification = classify_and_extract(full_input)

    expected_filters = {
        "query": "Deep Learning",
        "author": "Ilya Sutskever",
        "title": "Attention",
        "category": "cs.CL",
        "abstract": "NLP",
        "journal_ref": "NeurIPS",
        "doi": "10.1234/abcde",
        "exclude_words": "Computer Vision",
        "start_date": "20100101",
        "end_date": "20171231",
        "max_results": "1",
        "sort_by": "Relevance"
    }

    for key, value in expected_filters.items():
        actual = classification['filters'].get(key)
        expected = value

        if actual is None or expected is None:
            assert actual == expected
        else:
            assert actual.lower() == expected.lower()    

def test_future_date():
    future_query = 'Find me papers on AI from 2026'
    classification = classify_and_extract(future_query)
    
    expected_filters = {
        "query": 'AI',
        "author": None,
        "title": None,
        "category": None,
        "abstract": None,
        "journal_ref": None,
        "doi": None,
        "exclude_words": None,
        "start_date": None,
        "end_date": None,
        "max_results": '1',
        "sort_by": "relevance"
    }

    for key, value in expected_filters.items():
        actual = classification['filters'].get(key)
        expected = value

        if actual is None or expected is None:
            assert actual == expected
        else:
            assert actual.lower() == expected.lower()  