# lib/repositories/articles_repository.py
from typing import Any, Dict, Iterable, List, Optional, Tuple
from lib.db.mongo_client import get_db
from pymongo.collection import Collection


class ArticlesRepository:
    def __init__(self) -> None:
        self.collection: Collection = get_db()["articles"]

    def create_articles(self, data: Dict[str, Any]) -> str:
        result = self.collection.insert_one(data)
        return str(result.inserted_id)

    def aggregate_articles(self, pipeline: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        """
        Perform aggregation on the articles collection.
        :param pipeline: List of aggregation stages.
        :return: Iterable of documents resulting from the aggregation.
        """
        return self.collection.aggregate(pipeline)

    def get_articles(self, params: Dict[str, Any], projection: Optional[Dict[str, int]] = None):
        return self.collection.find(params, projection) if projection else self.collection.find(params)

    def get_distinct_samples(self, sample_str: str) -> List[str]:
        # case-insensitive suffix match for -YYYY-MM-DD
        regex_filter = {"sample": {"$regex": sample_str, "$options": "i"}}
        return list(self.collection.distinct("sample", regex_filter))

    def get_one_article(self, params: Dict[str, Any], sorting: Optional[List[Tuple[str, int]]] = None):
        return self.collection.find_one(params, sort=sorting) if sorting else self.collection.find_one(params)

    def update_articles(self, selector: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        result = self.collection.update_one(selector, update_data)
        return result.modified_count

    def delete_articles(self, selector: Dict[str, Any]) -> int:
        result = self.collection.delete_many(selector)
        return result.deleted_count

    def count_articles(self, params: Dict[str, Any]) -> int:
        return self.collection.count_documents(params)
    
    def get_articles_grouped_by_source(self) -> Dict[str, List[Dict[str, Any]]]:
        pipeline = [
            {
                "$group": {
                    "_id": "$source",
                    "articles": {"$push": "$$ROOT"}
                }
            }
        ]
        grouped_result = self.collection.aggregate(pipeline)
        grouped_dict = {}
        for group in grouped_result:
            source = group["_id"]
            articles = group["articles"]
            grouped_dict[source] = articles
        return grouped_dict

    def setup_indexes(self) -> None:
        name = self.collection.create_index([("isCleaned", 1), ("sample", 1)])
        print(f"✅ Compound index '{name}' created on 'isCleaned + sample'")

    def create_index(self, keys: List[Tuple[str, int]], **kwargs) -> str:
        """
        Create an index on the articles collection.
        :param keys: List of tuples specifying the fields and their sort order.
        :param kwargs: Additional options for index creation.
        :return: The name of the created index.
        """
        return self.collection.create_index(keys, **kwargs)
