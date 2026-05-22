import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_utils import *

standard_filters = {
    "query": "Deep Learning",
    "author": "Ilya Sutskever",
    "exclude_words": "Computer Vision",
    "end_date": "20171231",
    "max_results": '3',
    "sort_by": "relevance"
}  

small_filters = {"author": "Ilya Sutskever", "max_results": "10"}

full_filters = {
    "query": "Deep Learning",
    "author": "Ilya Sutskever",
    "title": "Attention",
    "category": "cs.CL",
    "abstract": "NLP",
    "journal_ref": "NeurIPS",
    "doi": "10.1234/abcde",
    "exclude_words": "Computer Vision",
    "start_date": "20100101",
    "end_date": "20171231"
}

def test_query_formatting():
    query = format_query(standard_filters)
    expected = 'Deep Learning AND au:"Ilya Sutskever" AND NOT "Computer Vision" AND submittedDate:[* TO 20171231]'
    assert query == expected

# Test format_query with minimal filters
def test_small_query_formatting():
    query = format_query(small_filters)
    expected = 'au:"Ilya Sutskever"'
    assert query == expected

# Test format_query with all fields
def test_full_query_formatting():    
    query = format_query(full_filters)
    expected = 'Deep Learning AND au:"Ilya Sutskever" AND ti:"Attention" AND cat:"cs.CL" AND abs:"NLP" AND jr:"NeurIPS" AND doi:"10.1234/abcde" AND NOT "Computer Vision" AND submittedDate:[20100101 TO 20171231]'

    assert query == expected

# Test fetch_papers to ensure that papers matching the query are fetched
def test_paper_fetching():
    papers = fetch_papers(standard_filters)

    for paper in papers:
        paper_title = paper.title
        paper_author_list = [author.name for author in paper.authors]
        paper_date = paper.published.strftime("%Y%m%d%H%M")

        assert any(standard_filters.get('author').lower() in a.lower() for a in paper_author_list)
        assert standard_filters.get('exclude_words').lower() not in paper_title.lower()
        assert (str(paper_date) <= str(standard_filters.get('end_date')))
    assert len(papers) == int(standard_filters.get('max_results'))

# Test for max results = 10 to see if every paper consistently matches the filters
def test_ten_fetching():
    papers = fetch_papers(small_filters)
    for paper in papers:
        paper_author_list = [author.name for author in paper.authors]

        assert any(standard_filters.get('author').lower() in a.lower() for a in paper_author_list)
    assert len(papers) == int(small_filters.get('max_results'))

# Test to see if extensive filters return results
def test_full_fetching():
    papers = fetch_papers(full_filters)
    for paper in papers:
        paper_title = paper.title
        paper_author_list = [author.name for author in paper.authors]
        paper_abstract = paper.summary
        paper_category = paper.primary_category
        paper_date = paper.published.strftime("%Y%m%d%H%M")

        assert any(full_filters.get('author', '').lower() in a.lower() for a in paper_author_list)
        assert full_filters.get('title', '').lower() in paper_title.lower()
        assert full_filters.get('category') in paper_category
        assert full_filters.get('abstract', '').lower() in paper_abstract.lower()
        assert full_filters.get('exclude_words', '').lower() not in paper_title.lower()
        assert (str(paper_date) >= str(full_filters.get('start_date', ""))) and (full_filters.get('end_date') is None or str(paper_date) <= str(full_filters.get('end_date', "999999999999")))

# Test fetch_papers with invalid query to ensure empty result
def test_fetching_no_results():
    bad_filters = {
        "query": "asdkfjasdlfj",  
        "max_results": 2,
        "sort_by": "relevance"
    }
    papers = fetch_papers(bad_filters)
    assert len(papers) == 0

def test_paper_chunking():
    chunk_size = 500

    papers = fetch_papers(standard_filters)
    paper_chunks = chunk_papers(papers)

    assert all(0 < len(chunk['text']) <= chunk_size for chunk in paper_chunks)

# Test chunk_papers skips non-PDF papers
def test_chunk_papers_skips_non_pdf():
    class DummyPaper:
        title = "Fake Paper"
        pdf_url = None 

    chunks = chunk_papers([DummyPaper()])
    assert chunks == [] 