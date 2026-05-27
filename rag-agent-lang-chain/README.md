# Build a RAG agent with LangChain

This repository is my process of following along "Build a RAG agent with LangChain", available online [here](https://docs.langchain.com/oss/python/langchain/rag)

- `agent.py` is a basic example as shown in the course.

- `agent_wikipedia_article.py` is an example that uses a wikipedia article instead of a blog post. This works pretty well, though does have the issue of including all HTML text in the document database, meaning information is technically diluted by nonsensical HTML elements. This could be fixed by filtering for text elements instead, in a similar way the blog post example does.

- `agent_web_search.py` is an example modified to have a web searching function, making use of duckduckgo's free web searching API. Bear in mind it will only work for popular or common content, as duckduckgo only returns JSON abstract text information for common searches. This can be expanded to dynamically search wikipedia for content as required if duckduckgo comes up empty.