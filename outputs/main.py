from lib.repositories.articles_repository import ArticlesRepository
from lib.repositories.metadata_repository import MetadataRepository
from lib.repositories.link_pool_repository import LinkPoolRepository



def articles():
    repo_articles = ArticlesRepository()
    articles = repo_articles.get_articles({})
    for article in articles:
        print(article)
        print("----")


def access_metadata():
    repo_metadat = MetadataRepository()
    docs = repo_metadat.get_metadata({"_id": "1-2025-09-17"})
    for doc in docs:
        print(doc)
        print("----")
def delete_metadata():
    repo_metadat = MetadataRepository()
    result = repo_metadat.delete_metadata_one({"_id": "1-2025-09-17"})
    print(f"Deleted {result} documents.")

def get_links():
    repo = LinkPoolRepository()
    links = repo.get_link({})
    for link in links:
        print(link)
        print("----")

def getAllArticlesAndEdit():
    repo = ArticlesRepository()
    articles = repo.get_articles({})
    for article in articles:
        print(article)
        print("----")
        repo.update_articles({"_id": article["_id"]}, {"$set": {"relevanceStatus": "pending"}})
        print(f"Updated article {article['_id']} to set relevanceStatus to pending")

def countArticles():
    repo = ArticlesRepository()
    count = repo.count_articles({})
    print(f"Total articles count: {count}")

def articles_documents_grouped_by_source():
    repo = ArticlesRepository()
    grouped_articles = repo.get_articles_grouped_by_source()
    for source, articles in grouped_articles.items():
        print(f"Source: {source}")
        print("----")

if __name__ == "__main__":
    articles_documents_grouped_by_source()
