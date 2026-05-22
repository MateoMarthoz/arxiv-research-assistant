import arxiv
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


"""
Description:
This script searches for papers on ArXiv based on given filters, checks if results match the criteria,
downloads their PDFs, extracts and splits their content into chunks with metadata (title, author, date).
"""


def format_query(filters):
    """ Builds an ArXiv API query string from filter dictionary. """

    search_parts = []

    if filters.get('query'):
        search_parts.append(filters.get('query'))  # Base query
        
    if filters.get('author'):
        search_parts.append(f'au:"{filters["author"]}"')  
    if filters.get('title'):
        search_parts.append(f'ti:"{filters["title"]}"')  
    if filters.get('category'):
        search_parts.append(f'cat:"{filters["category"]}"')  
    if filters.get('abstract'):
        search_parts.append(f'abs:"{filters["abstract"]}"')  
    if filters.get('journal_ref'):
        search_parts.append(f'jr:"{filters["journal_ref"]}"')  
    if filters.get('doi'):
        search_parts.append(f'doi:"{filters["doi"]}"')  
    if filters.get('exclude_words'):
        search_parts.append(f'NOT "{filters["exclude_words"]}"')  
    start = filters.get('start_date') or '*'
    end = filters.get('end_date') or '99999999'
    if filters.get('start_date') is not None or filters.get('end_date') is not None:
        search_parts.append(f'submittedDate:[{start} TO {end}]')
    
    if search_parts:
        query_str = " AND ".join(search_parts)  # Combine with AND
    else:
        # If query is not detected, return 0 and skip paper search part
        print("Query not detected")
        return 0
    return query_str

def fetch_papers(filters):
    """ Fetches academic papers from ArXiv based on filters. """
    
    search_query = format_query(filters)

    # Create ArXiv search object
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=int(filters.get('max_results', 1)),
        sort_by=arxiv.SortCriterion.Relevance if filters.get('sort_by') == "relevance" else arxiv.SortCriterion.SubmittedDate
    )

    results = list(client.results(search))  
    return results

# a global variable that stores titles of paper retrieved
stored_paper_titles = set()

def verify_results(results, filters):
    """ Verifies if fetched papers match the filters and prints details. """
    verified_results = []  # Store matching papers

    for result in results:
        paper_title = result.title

        # If paper has been retrieved before, skip the paper
        if paper_title in stored_paper_titles:
            print(f"Skipping '{paper_title}' — Already stored.")
            continue    

        paper_title = result.title
        paper_author_list = [author.name for author in result.authors]
        paper_abstract = result.summary
        paper_category = result.primary_category
        paper_date = result.published.strftime("%Y%m%d%H%M")

        # Check filter matches
        match_results = {
            "Author Match": filters.get('author') is None or any(filters.get('author', '').lower() in a.lower() for a in paper_author_list),
            # "Title Match": filters.get('title') is None or filters.get('title', '').lower() in paper_title.lower(),  # commented out to allow vague search for title
            "Category Match": filters.get('category') is None or filters.get('category') in paper_category,
            "Abstract Match": filters.get('abstract') is None or filters.get('abstract', '').lower() in paper_abstract.lower(),
            "Exclusion (NOT)": filters.get('exclude_words') is None or filters.get('exclude_words', '').lower() not in paper_title.lower(),
            "Date Range Match": (
                filters.get('start_date') is None or str(paper_date) >= str(filters.get('start_date', ""))
            ) and (
                filters.get('end_date') is None or str(paper_date) <= str(filters.get('end_date', "999999999999"))
            )
        }

        # Add paper if it meets all criteria
        if all(match_results.values()):
            verified_results.append(result)
            stored_paper_titles.add(paper_title)
            print('added ',paper_title,' to prevent searching it next time')

    return verified_results


def chunk_papers(results):
    """ Downloads, extracts, and chunks text from ArXiv PDFs, skipping papers without PDFs. """
    all_chunks = []

    for result in results:
        if not result.pdf_url:
            print(f"Skipping '{result.title}' — No PDF available.")
            continue

        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            try:
                temp_pdf.close()
                result.download_pdf(filename=temp_pdf.name)
                loader = PyPDFLoader(temp_pdf.name)
                papers = loader.load()
                # Combine the text from all pages
                full_text = " ".join(p.page_content for p in papers)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
                paper_chunks = text_splitter.create_documents([full_text])

                paper_title = result.title
                paper_author_list = [author.name for author in result.authors] 
                paper_date = result.published.strftime("%Y%m%d")

                # Tag each chunk with paper metadata and store content under "text"
                for i, chunk in enumerate(paper_chunks):
                    content = getattr(chunk, "page_content", "")
                    all_chunks.append({
                        "text": content,
                        "title":paper_title,
                        "author": paper_author_list,
                        "date": paper_date,
                        "chunk_index": i
                    })

                print(f"Extracted {len(paper_chunks)} chunks from '{result.title}'")

            except Exception as e:
                print(f"Error processing '{result.title}': {e}")
                time.sleep(5)

    print(f"\nTotal Chunks Extracted: {len(all_chunks)}\n")
    return all_chunks

def clear():
    # this function clears the retrieved paper history
    global stored_paper_titles
    stored_paper_titles.clear()
