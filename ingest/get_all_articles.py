# get_all_articles.py

from ingest.custom_scrapers import scrape_bbc_stream, scrape_cnn_stream, scrape_wsj_stream, scrape_aljazeera, scrape_dw_stream, scrape_guardian_stream, scrape_reuters_stream, scrape_guardian_selenium_stream, scrape_france24_selenium_stream, scrape_npr_selenium_stream 


def get_all_articles():
    seen_urls = set()
    unique_articles = []

    for scrape_func in [scrape_bbc_stream, scrape_cnn_stream,
                        scrape_wsj_stream, scrape_aljazeera, scrape_dw_stream, scrape_guardian_stream, scrape_reuters_stream, scrape_guardian_selenium_stream, scrape_france24_selenium_stream, scrape_npr_selenium_stream]:
        articles = scrape_func()
        for article in articles:
            url = article.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
    print(f"[INFO] Total articles fetched: {len(unique_articles)}")
    return unique_articles
