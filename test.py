from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()


web_search = TavilySearch(max_results=3)

results = web_search.invoke("latest RBI KYC circular 2026")
print(results.keys())
