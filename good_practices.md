# Guía de Buenas Prácticas y Mejoras Arquitectónicas

## 📚 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Análisis Comparativo de Arquitecturas](#análisis-comparativo-de-arquitecturas)
3. [Estructura de Directorios](#estructura-de-directorios)
4. [Configuración y Variables de Entorno](#configuración-y-variables-de-entorno)
5. [Separación de Responsabilidades](#separación-de-responsabilidades)
6. [Manejo de Estado y Persistencia](#manejo-de-estado-y-persistencia)
7. [Logging y Manejo de Errores](#logging-y-manejo-de-errores)
8. [Gestión de Dependencias](#gestión-de-dependencias)
9. [Testing y Calidad de Código](#testing-y-calidad-de-código)
10. [CLI y Puntos de Entrada](#cli-y-puntos-de-entrada)
11. [Documentación](#documentación)
12. [Plan de Migración Paso a Paso](#plan-de-migración-paso-a-paso)
13. [Checklist de Implementación](#checklist-de-implementación)

---

## 1. Resumen Ejecutivo

### 🎯 Objetivo

Transformar **news-crawler-ai** de un script monolítico de scraping a una aplicación modular, mantenible y escalable siguiendo los patrones arquitectónicos demostrados en **thread_events_agents**.

### 📊 Diferencias Clave

| Aspecto | news-crawler-ai (Actual) | thread_events_agents (Referencia) | Impacto |
|---------|-------------------------|----------------------------------|---------|
| **Estructura** | Flat (`ingest/`, `lib/`) | Modular (`core/`, `providers/`, `repositories/`) | 🔴 Alto |
| **Configuración** | Variables dispersas en código | Dataclasses centralizadas con `dotenv` | 🟡 Medio |
| **Estado** | Sin persistencia entre ejecuciones | `StateManager` con JSON | 🔴 Alto |
| **Logging** | `print()` inconsistente | `logging` estructurado + MongoDB | 🔴 Alto |
| **Testing** | Inexistente | pytest + coverage | 🔴 Alto |
| **CLI** | Scripts separados | CLI unificado con subcomandos | 🟡 Medio |
| **Dependencias** | `requirements.txt` básico | `pyproject.toml` + Poetry | 🟡 Medio |
| **Documentación** | README básico | README + diagramas + docstrings | 🟡 Medio |

---

## 2. Análisis Comparativo de Arquitecturas

### Estado Actual: news-crawler-ai

```
news-crawler-ai/
├── ingest/                    # Todo mezclado: scraping, API, clasificación
│   ├── classifier.py          # Orquestación + clasificación (múltiples responsabilidades)
│   ├── get_all_articles.py    # Agregador de scrapers
│   ├── custom_scrapers.py     # 10+ scrapers en un solo archivo
│   ├── call_to_genai.py       # Llamadas a Gemini (provider + lógica)
│   ├── call_to_webhook.py     # Webhooks (provider)
│   ├── summarizer.py          # Summarización
│   └── utils.py               # Utilidades genéricas
│
├── lib/
│   ├── db/
│   │   └── mongo_client.py    # Cliente MongoDB
│   └── repositories/          # ✅ Patrón correcto
│       ├── articles_repository.py
│       ├── link_pool_repository.py
│       └── metadata_repository.py
│
├── scripts/
│   └── bootstrap_models.py
│
└── main.py                    # Placeholder vacío
```

**Problemas identificados:**

1. **Alta cohesión en `ingest/`**: Mezcla scraping, procesamiento, clasificación y orquestación
2. **No hay capa `core/`**: Orquestación dispersa en `classifier.py`
3. **No hay capa `providers/`**: Proveedores externos (Gemini, LLM, webhooks) mezclados con lógica
4. **Sin persistencia de estado**: No se puede reanudar si falla
5. **Logging primitivo**: Solo `print()`, sin auditoría
6. **Testing ausente**: No hay tests
7. **Configuración hardcodeada**: Variables de entorno leídas directamente en múltiples archivos

---

### Arquitectura Objetivo: Inspirada en thread_events_agents

```
news-crawler-ai/
├── app/                       # ⭐ NUEVO: API FastAPI (opcional, futuro)
│   ├── __init__.py
│   ├── main.py               # Punto de entrada API
│   ├── articles.py           # Endpoints CRUD artículos
│   ├── webhooks.py           # Webhooks de integración
│   └── logs.py               # Endpoints de auditoría
│
├── news_crawler/              # ⭐ RENOMBRAR: de ingest/ a news_crawler/
│   ├── __init__.py
│   │
│   ├── core/                 # ⭐ NUEVO: Orquestación principal
│   │   ├── __init__.py
│   │   ├── config.py         # ⭐ Configuración centralizada con dataclasses
│   │   ├── state.py          # ⭐ Persistencia de estado (StateManager)
│   │   ├── orchestrator.py   # ⭐ Pipeline principal (extrae de classifier.py)
│   │   └── scraper_manager.py # ⭐ Coordinador de scrapers
│   │
│   ├── providers/            # ⭐ NUEVO: Servicios externos
│   │   ├── __init__.py
│   │   ├── genai_provider.py # ⭐ Extrae de call_to_genai.py
│   │   ├── webhook_provider.py # ⭐ Extrae de call_to_webhook.py
│   │   ├── newsapi_provider.py # ⭐ Extrae de news_api_scrapper.py
│   │   └── ollama_provider.py  # ⭐ Extrae de call_to_ollama.py
│   │
│   ├── scrapers/             # ⭐ NUEVO: Scrapers organizados
│   │   ├── __init__.py
│   │   ├── base_scraper.py   # ⭐ Clase abstracta base
│   │   ├── bbc_scraper.py    # ⭐ Separar de custom_scrapers.py
│   │   ├── cnn_scraper.py
│   │   ├── wsj_scraper.py
│   │   ├── dw_scraper.py     # ⭐ Mover crawler_dw.py aquí
│   │   └── ...               # ⭐ Un archivo por scraper
│   │
│   ├── processors/           # ⭐ NUEVO: Procesadores de texto
│   │   ├── __init__.py
│   │   ├── classifier.py     # ⭐ Solo clasificación (extraer de classifier.py)
│   │   ├── summarizer.py     # ⭐ Mover summarizer.py
│   │   ├── sentiment_analyzer.py # ⭐ Separar de classifier.py
│   │   └── entity_extractor.py   # ⭐ Lógica de extracción
│   │
│   ├── repositories/         # ✅ YA EXISTE (mover de lib/)
│   │   ├── __init__.py
│   │   ├── articles_repository.py
│   │   ├── link_pool_repository.py
│   │   ├── metadata_repository.py
│   │   └── pipeline_logs_repository.py # ⭐ NUEVO
│   │
│   ├── db/                   # ✅ Mover de lib/db/
│   │   ├── __init__.py
│   │   └── mongo_client.py
│   │
│   └── utils/                # ✅ YA EXISTE
│       ├── __init__.py
│       ├── text.py           # ⭐ Normalización texto
│       ├── validators.py     # ⭐ Validaciones
│       └── logging.py        # ⭐ NUEVO: Configuración logging
│
├── tests/                    # ⭐ NUEVO: Tests
│   ├── __init__.py
│   ├── test_scrapers.py
│   ├── test_processors.py
│   ├── test_repositories.py
│   └── test_providers.py
│
├── scripts/                  # ✅ YA EXISTE
│   ├── __init__.py
│   ├── bootstrap_models.py
│   └── systemd/
│       ├── news-crawler-ai.service
│       └── news-crawler-ai.timer
│
├── pyproject.toml            # ⭐ NUEVO: Gestión con Poetry
├── .env.example              # ⭐ NUEVO: Template de configuración
├── .gitignore
├── README.md                 # ✅ Mejorar con estructura de referencia
└── good_practices.md         # Este documento
```

---

## 3. Estructura de Directorios

### 3.1 Crear `news_crawler/core/`

**Propósito**: Orquestación principal y configuración centralizada.

**Archivos a crear:**

#### `news_crawler/core/config.py`

```python
"""Configuración centralizada del sistema."""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MongoConfig:
    """Configuración de MongoDB."""
    uri: str
    db_name: str
    articles_collection: str = "articles"
    link_pool_collection: str = "link_pool"
    metadata_collection: str = "metadata"
    logs_collection: str = "pipeline_logs"
    
    @classmethod
    def from_env(cls) -> "MongoConfig":
        return cls(
            uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            db_name=os.getenv("MONGODB_DB", "news_feeder_db"),
            articles_collection=os.getenv("MONGO_ARTICLES_COLLECTION", "articles"),
            link_pool_collection=os.getenv("MONGO_LINK_POOL_COLLECTION", "link_pool"),
            metadata_collection=os.getenv("MONGO_METADATA_COLLECTION", "metadata"),
            logs_collection=os.getenv("MONGO_LOGS_COLLECTION", "pipeline_logs"),
        )

@dataclass
class GenAIConfig:
    """Configuración de Google Generative AI."""
    api_key: str
    model: str = "gemini-2.0-flash"
    temperature: float = 0.0
    top_p: float = 1.0
    max_chunk_chars: int = 12000
    
    @classmethod
    def from_env(cls) -> "GenAIConfig":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set")
        
        return cls(
            api_key=api_key,
            model=os.getenv("GENAI_MODEL", "gemini-2.0-flash"),
            temperature=float(os.getenv("GENAI_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("GENAI_TOP_P", "1.0")),
            max_chunk_chars=int(os.getenv("GENAI_MAX_CHUNK_CHARS", "12000")),
        )

@dataclass
class WebhookConfig:
    """Configuración de webhooks."""
    embedding_url: Optional[str] = None
    thread_events_url: Optional[str] = None
    signature: str = ""
    timeout: int = 60
    max_retries: int = 3
    
    @classmethod
    def from_env(cls) -> "WebhookConfig":
        return cls(
            embedding_url=os.getenv("WEBHOOK_URL"),
            thread_events_url=os.getenv("WEBHOOK_URL_THREAD_EVENTS"),
            signature=os.getenv("WEBHOOK_SIGNATURE", ""),
            timeout=int(os.getenv("WEBHOOK_TIMEOUT", "60")),
            max_retries=int(os.getenv("WEBHOOK_MAX_RETRIES", "3")),
        )

@dataclass
class AppConfig:
    """Configuración general de la aplicación."""
    app_name: str = "news-crawler-ai"
    log_level: str = "INFO"
    transformers_cache: str = "./models/transformers"
    enable_webhooks: bool = True
    enable_genai: bool = True
    news_fetch_timeout: int = 20
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            app_name=os.getenv("APP_NAME", "news-crawler-ai"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            transformers_cache=os.getenv("TRANSFORMERS_CACHE", "./models/transformers"),
            enable_webhooks=os.getenv("ENABLE_WEBHOOKS", "1") == "1",
            enable_genai=os.getenv("ENABLE_GENAI", "1") == "1",
            news_fetch_timeout=int(os.getenv("NEWS_FETCH_TIMEOUT", "20")),
        )

# Instancias globales de configuración
MONGO_CONFIG = MongoConfig.from_env()
GENAI_CONFIG = GenAIConfig.from_env() if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") else None
WEBHOOK_CONFIG = WebhookConfig.from_env()
APP_CONFIG = AppConfig.from_env()

# Validación al importar
def validate_config():
    """Valida que la configuración sea correcta."""
    errors = []
    
    if not MONGO_CONFIG.uri:
        errors.append("MONGO_URI is required")
    
    if APP_CONFIG.enable_genai and not GENAI_CONFIG:
        errors.append("GEMINI_API_KEY or GOOGLE_API_KEY required when ENABLE_GENAI=1")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Validar al importar
validate_config()
```

#### `news_crawler/core/state.py`

```python
"""Gestión de estado persistente del pipeline."""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StateManager:
    """Gestiona el estado del procesamiento para permitir reanudación."""
    
    def __init__(self, state_file: str = "crawler_state.json"):
        self.state_file = Path(state_file)
        self._state: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> Dict[str, Any]:
        """Carga el estado desde archivo JSON."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
                logger.info(f"State loaded from {self.state_file}")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                self._state = {}
        else:
            logger.info("No existing state file, starting fresh")
            self._state = {}
        return self._state
    
    def save(self) -> None:
        """Guarda el estado a archivo JSON."""
        try:
            # Crear directorio si no existe
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, default=str)
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def get_last_batch_id(self) -> Optional[str]:
        """Obtiene el último batch_id procesado."""
        return self._state.get("last_batch_id")
    
    def set_last_batch_id(self, batch_id: str) -> None:
        """Establece el último batch_id procesado."""
        self._state["last_batch_id"] = batch_id
        self._state["last_updated"] = datetime.utcnow().isoformat()
        self.save()
    
    def get_scraper_state(self, scraper_name: str) -> Dict[str, Any]:
        """Obtiene el estado de un scraper específico."""
        scrapers = self._state.get("scrapers", {})
        return scrapers.get(scraper_name, {})
    
    def set_scraper_state(
        self,
        scraper_name: str,
        last_url: Optional[str] = None,
        articles_processed: Optional[int] = None,
        last_run: Optional[str] = None
    ) -> None:
        """Actualiza el estado de un scraper."""
        if "scrapers" not in self._state:
            self._state["scrapers"] = {}
        
        if scraper_name not in self._state["scrapers"]:
            self._state["scrapers"][scraper_name] = {}
        
        scraper = self._state["scrapers"][scraper_name]
        
        if last_url is not None:
            scraper["last_url"] = last_url
        if articles_processed is not None:
            scraper["articles_processed"] = articles_processed
        if last_run is not None:
            scraper["last_run"] = last_run
        
        scraper["updated_at"] = datetime.utcnow().isoformat()
        self.save()
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del pipeline."""
        return self._state.get("stats", {
            "total_articles_processed": 0,
            "total_batches": 0,
            "last_run": None
        })
    
    def increment_stats(self, articles_count: int) -> None:
        """Incrementa estadísticas del pipeline."""
        if "stats" not in self._state:
            self._state["stats"] = {
                "total_articles_processed": 0,
                "total_batches": 0,
                "last_run": None
            }
        
        self._state["stats"]["total_articles_processed"] += articles_count
        self._state["stats"]["total_batches"] += 1
        self._state["stats"]["last_run"] = datetime.utcnow().isoformat()
        self.save()
    
    def reset(self, scraper_name: Optional[str] = None) -> None:
        """Resetea el estado para un scraper o todo."""
        if scraper_name:
            if "scrapers" in self._state and scraper_name in self._state["scrapers"]:
                del self._state["scrapers"][scraper_name]
                logger.info(f"State reset for scraper: {scraper_name}")
        else:
            self._state = {}
            logger.info("All state reset")
        self.save()
```

#### `news_crawler/core/orchestrator.py`

```python
"""Orquestador principal del pipeline de scraping y clasificación."""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from news_crawler.core.config import APP_CONFIG, WEBHOOK_CONFIG
from news_crawler.core.state import StateManager
from news_crawler.scrapers import get_all_articles
from news_crawler.processors.classifier import classify_article
from news_crawler.processors.summarizer import smart_summarize
from news_crawler.providers.genai_provider import GenAIProvider
from news_crawler.providers.webhook_provider import WebhookProvider
from news_crawler.repositories.articles_repository import ArticlesRepository
from news_crawler.repositories.link_pool_repository import LinkPoolRepository
from news_crawler.repositories.metadata_repository import MetadataRepository

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orquesta todo el flujo de scraping, procesamiento y almacenamiento."""
    
    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        dry_run: bool = False
    ):
        self.state_manager = state_manager or StateManager()
        self.dry_run = dry_run
        
        # Repositorios
        self.articles_repo = ArticlesRepository()
        self.link_pool_repo = LinkPoolRepository()
        self.metadata_repo = MetadataRepository()
        
        # Providers
        self.genai = GenAIProvider() if APP_CONFIG.enable_genai else None
        self.webhook = WebhookProvider() if APP_CONFIG.enable_webhooks else None
        
        logger.info(f"PipelineOrchestrator initialized (dry_run={dry_run})")
    
    def run(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de scraping y clasificación.
        
        Args:
            limit: Número máximo de artículos a procesar (None = sin límite)
        
        Returns:
            Dict con estadísticas de la ejecución
        """
        batch_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Starting pipeline run (batch_id={batch_id})")
        
        stats = {
            "batch_id": batch_id,
            "started_at": start_time.isoformat(),
            "articles_processed": 0,
            "articles_failed": 0,
            "articles_skipped": 0,
            "scrapers_used": set(),
        }
        
        try:
            # 1. Obtener artículos de todos los scrapers
            articles_generator = get_all_articles()
            
            # 2. Procesar cada artículo
            for idx, article in enumerate(articles_generator):
                if limit and idx >= limit:
                    logger.info(f"Reached limit of {limit} articles")
                    break
                
                try:
                    result = self._process_article(article, batch_id)
                    
                    if result["status"] == "success":
                        stats["articles_processed"] += 1
                    elif result["status"] == "skipped":
                        stats["articles_skipped"] += 1
                    else:
                        stats["articles_failed"] += 1
                    
                    stats["scrapers_used"].add(article.get("source", "unknown"))
                    
                except Exception as e:
                    logger.error(f"Error processing article: {e}", exc_info=True)
                    stats["articles_failed"] += 1
            
            # 3. Guardar metadata del batch
            end_time = datetime.utcnow()
            stats["finished_at"] = end_time.isoformat()
            stats["duration_seconds"] = (end_time - start_time).total_seconds()
            stats["scrapers_used"] = list(stats["scrapers_used"])
            
            if not self.dry_run:
                self._save_batch_metadata(batch_id, stats)
                self.state_manager.set_last_batch_id(batch_id)
                self.state_manager.increment_stats(stats["articles_processed"])
            
            logger.info(
                f"Pipeline run completed: {stats['articles_processed']} processed, "
                f"{stats['articles_skipped']} skipped, {stats['articles_failed']} failed"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Pipeline run failed: {e}", exc_info=True)
            raise
    
    def _process_article(
        self,
        article: Dict[str, Any],
        batch_id: str
    ) -> Dict[str, str]:
        """
        Procesa un solo artículo a través del pipeline completo.
        
        Returns:
            Dict con status: success/skipped/failed
        """
        url = article.get("url", "")
        
        # 1. Check si ya fue procesado
        if self.link_pool_repo.is_processed(url):
            logger.debug(f"Skipping already processed: {url}")
            return {"status": "skipped", "reason": "already_processed"}
        
        # 2. Limpiar texto y extraer entidades (si GenAI está disponible)
        if self.genai:
            cleaned_data = self.genai.clean_and_extract_entities(article.get("text", ""))
            article["text"] = cleaned_data.get("cleaned_text", article["text"])
            article["locations"] = cleaned_data.get("locations", [])
            article["organizations"] = cleaned_data.get("organizations", [])
            article["persons"] = cleaned_data.get("persons", [])
        
        # 3. Summarizar si es necesario
        article["summary"] = smart_summarize(article.get("text", ""))
        
        # 4. Clasificar (sentiment + topic)
        classification = classify_article(article.get("text", ""))
        article.update(classification)
        
        # 5. Guardar en MongoDB
        if not self.dry_run:
            article["sample"] = batch_id
            article["scraped_at"] = article.get("scraped_at", datetime.utcnow().isoformat())
            article_id = self.articles_repo.insert_article(article)
            
            # 6. Marcar URL como procesada
            self.link_pool_repo.mark_as_processed(url, batch_id)
            
            # 7. Enviar webhooks
            if self.webhook:
                self.webhook.send_article_webhooks(article_id, article)
        
        logger.info(f"✅ Processed: {article.get('title', 'Untitled')[:60]}...")
        return {"status": "success"}
    
    def _save_batch_metadata(self, batch_id: str, stats: Dict[str, Any]) -> None:
        """Guarda metadata del batch en MongoDB."""
        try:
            self.metadata_repo.insert_metadata(batch_id, stats)
            logger.debug(f"Batch metadata saved: {batch_id}")
        except Exception as e:
            logger.error(f"Error saving batch metadata: {e}")
```

---

### 3.2 Crear `news_crawler/providers/`

**Propósito**: Aislar interacciones con servicios externos (APIs, webhooks, LLMs).

#### `news_crawler/providers/genai_provider.py`

```python
"""Proveedor de Google Generative AI."""
import logging
from typing import Dict, Any
import google.generativeai as genai

from news_crawler.core.config import GENAI_CONFIG

logger = logging.getLogger(__name__)

class GenAIProvider:
    """Proveedor para Google Generative AI (Gemini)."""
    
    def __init__(self):
        if not GENAI_CONFIG:
            raise ValueError("GenAI configuration not available")
        
        genai.configure(api_key=GENAI_CONFIG.api_key)
        self.model = genai.GenerativeModel(
            GENAI_CONFIG.model,
            generation_config={
                "temperature": GENAI_CONFIG.temperature,
                "top_p": GENAI_CONFIG.top_p,
            }
        )
        logger.info(f"GenAIProvider initialized with model: {GENAI_CONFIG.model}")
    
    def clean_and_extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Limpia texto y extrae entidades nombradas.
        
        Args:
            text: Texto crudo a procesar
        
        Returns:
            Dict con cleaned_text, locations, organizations, persons
        """
        if not text or len(text.strip()) < 50:
            return {
                "cleaned_text": text,
                "locations": [],
                "organizations": [],
                "persons": []
            }
        
        # Truncar si es muy largo
        text_chunk = text[:GENAI_CONFIG.max_chunk_chars]
        
        prompt = f"""
You are a text cleaning and entity extraction assistant.

Given the following article text, perform TWO tasks:

1. CLEAN the text by removing:
   - Advertisements and promotional content
   - Cookie notices and privacy banners
   - Navigation elements and menus
   - Author bylines (unless part of the story)
   - Social media sharing buttons text
   - "Read more" or "Subscribe" calls to action
   - HTML artifacts or formatting codes

2. EXTRACT named entities:
   - locations (cities, regions, countries)
   - organizations (companies, institutions, agencies)
   - persons (full names or last names mentioned)

Return ONLY valid JSON with this structure:
{{
  "cleaned_text": "The cleaned article text...",
  "locations": ["London", "United Kingdom"],
  "organizations": ["BBC", "Parliament"],
  "persons": ["Johnson", "Biden"]
}}

Article text:
{text_chunk}
"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Parsear JSON
            import json
            # Limpiar markdown si existe
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(result_text)
            
            # Validar estructura
            result.setdefault("cleaned_text", text)
            result.setdefault("locations", [])
            result.setdefault("organizations", [])
            result.setdefault("persons", [])
            
            return result
            
        except Exception as e:
            logger.error(f"GenAI call failed: {e}")
            return {
                "cleaned_text": text,
                "locations": [],
                "organizations": [],
                "persons": []
            }
```

#### `news_crawler/providers/webhook_provider.py`

```python
"""Proveedor de webhooks."""
import logging
import hashlib
import hmac
import json
import time
from typing import Dict, Any, Optional
import requests

from news_crawler.core.config import WEBHOOK_CONFIG

logger = logging.getLogger(__name__)

class WebhookProvider:
    """Proveedor para envío de webhooks con reintentos."""
    
    def __init__(self):
        self.config = WEBHOOK_CONFIG
        logger.info("WebhookProvider initialized")
    
    def send_article_webhooks(
        self,
        article_id: str,
        article: Dict[str, Any]
    ) -> None:
        """
        Envía webhooks para un artículo procesado.
        
        Args:
            article_id: ID del artículo en MongoDB
            article: Datos completos del artículo
        """
        # Webhook de embedding (payload completo)
        if self.config.embedding_url:
            payload = {
                "article_id": str(article_id),
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "text": article.get("text", ""),
                "topic": article.get("topic", ""),
                "source": article.get("source", ""),
                "sentiment": article.get("sentiment", {}),
                "scraped_at": article.get("scraped_at", "")
            }
            self._send_webhook(self.config.embedding_url, payload, "embedding")
        
        # Webhook de thread-events (payload ligero)
        if self.config.thread_events_url:
            payload = {
                "article_id": str(article_id),
                "source": article.get("source", ""),
                "scraped_at": article.get("scraped_at", "")
            }
            self._send_webhook(self.config.thread_events_url, payload, "thread-events")
    
    def _send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        webhook_name: str
    ) -> bool:
        """
        Envía un webhook con reintentos y firma.
        
        Returns:
            True si exitoso, False si falla
        """
        payload_json = json.dumps(payload, default=str)
        
        # Generar firma HMAC
        signature = ""
        if self.config.signature:
            signature = hmac.new(
                self.config.signature.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
        
        headers = {
            "Content-Type": "application/json",
            "X-Signature": signature
        }
        
        # Reintentos con backoff exponencial
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url,
                    data=payload_json,
                    headers=headers,
                    timeout=self.config.timeout
                )
                
                if response.status_code in [200, 201, 202]:
                    logger.debug(f"Webhook {webhook_name} sent successfully")
                    return True
                elif response.status_code in [429, 500, 502, 503, 504]:
                    # Códigos que justifican reintento
                    wait_time = 2 ** attempt  # Backoff exponencial
                    logger.warning(
                        f"Webhook {webhook_name} failed with {response.status_code}, "
                        f"retrying in {wait_time}s (attempt {attempt + 1}/{self.config.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Webhook {webhook_name} failed with {response.status_code}: {response.text}")
                    return False
                    
            except requests.exceptions.Timeout:
                logger.error(f"Webhook {webhook_name} timed out (attempt {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Webhook {webhook_name} error: {e}")
                return False
        
        logger.error(f"Webhook {webhook_name} failed after {self.config.max_retries} attempts")
        return False
```

---

### 3.3 Crear `news_crawler/scrapers/`

**Propósito**: Organizar scrapers con una interfaz común.

#### `news_crawler/scrapers/base_scraper.py`

```python
"""Clase base abstracta para todos los scrapers."""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    """Clase base para todos los scrapers de noticias."""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        logger.info(f"Scraper initialized: {source_name}")
    
    @abstractmethod
    def scrape(self) -> Iterator[Dict[str, Any]]:
        """
        Método abstracto que debe implementar cada scraper.
        
        Yields:
            Dict con keys: title, url, text, source, scraped_at
        """
        pass
    
    def _validate_article(self, article: Dict[str, Any]) -> bool:
        """
        Valida que un artículo tenga los campos mínimos requeridos.
        
        Returns:
            True si es válido, False si no
        """
        required_fields = ["title", "url", "text", "source"]
        
        for field in required_fields:
            if not article.get(field):
                logger.warning(f"Article missing required field: {field}")
                return False
        
        # Validar longitud mínima de texto
        if len(article["text"].strip()) < 100:
            logger.warning(f"Article text too short: {len(article['text'])} chars")
            return False
        
        return True
```

**Ejemplo de implementación:** `news_crawler/scrapers/bbc_scraper.py`

```python
"""Scraper para BBC News."""
import feedparser
from datetime import datetime
from typing import Iterator, Dict, Any

from news_crawler.scrapers.base_scraper import BaseScraper
from news_crawler.utils.text import fetch_and_extract

class BBCScraper(BaseScraper):
    """Scraper para BBC News RSS."""
    
    RSS_FEEDS = [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "http://feeds.bbci.co.uk/news/uk/rss.xml",
        "http://feeds.bbci.co.uk/news/business/rss.xml"
    ]
    
    def __init__(self):
        super().__init__("bbc-news")
    
    def scrape(self) -> Iterator[Dict[str, Any]]:
        """Scrapea artículos de todos los feeds RSS de BBC."""
        for feed_url in self.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    url = entry.get("link", "")
                    if not url:
                        continue
                    
                    # Extraer texto completo
                    text = fetch_and_extract(url)
                    if not text:
                        continue
                    
                    article = {
                        "title": entry.get("title", ""),
                        "url": url,
                        "text": text,
                        "source": self.source_name,
                        "scraped_at": datetime.utcnow().isoformat()
                    }
                    
                    if self._validate_article(article):
                        yield article
                        
            except Exception as e:
                logger.error(f"Error scraping BBC feed {feed_url}: {e}")
```

#### `news_crawler/scrapers/__init__.py`

```python
"""Agregador de todos los scrapers."""
from typing import Iterator, Dict, Any
import logging

from news_crawler.scrapers.bbc_scraper import BBCScraper
from news_crawler.scrapers.cnn_scraper import CNNScraper
# ... importar otros scrapers

logger = logging.getLogger(__name__)

# Registro de scrapers activos
ACTIVE_SCRAPERS = [
    BBCScraper,
    CNNScraper,
    # ... agregar otros
]

def get_all_articles() -> Iterator[Dict[str, Any]]:
    """
    Agrega artículos de todos los scrapers activos.
    
    Yields:
        Dict con article data
    """
    for scraper_class in ACTIVE_SCRAPERS:
        try:
            scraper = scraper_class()
            logger.info(f"Starting scraper: {scraper.source_name}")
            
            for article in scraper.scrape():
                yield article
                
        except Exception as e:
            logger.error(f"Scraper {scraper_class.__name__} failed: {e}", exc_info=True)
```

---

## 4. Configuración y Variables de Entorno

### 4.1 Crear `.env.example`

```env
# =============================================================================
# NEWS-CRAWLER-AI CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# MongoDB Configuration
# -----------------------------------------------------------------------------
MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB=news_feeder_db

# Optional: Override collection names
MONGO_ARTICLES_COLLECTION=articles
MONGO_LINK_POOL_COLLECTION=link_pool
MONGO_METADATA_COLLECTION=metadata
MONGO_LOGS_COLLECTION=pipeline_logs

# -----------------------------------------------------------------------------
# Google Generative AI (Gemini)
# -----------------------------------------------------------------------------
GEMINI_API_KEY=your-gemini-api-key-here
# Alternatively: GOOGLE_API_KEY=your-google-api-key-here

# GenAI model settings
GENAI_MODEL=gemini-2.0-flash
GENAI_TEMPERATURE=0.0
GENAI_TOP_P=1.0
GENAI_MAX_CHUNK_CHARS=12000

# -----------------------------------------------------------------------------
# Webhook Integration
# -----------------------------------------------------------------------------
# Full article payload for embedding service
WEBHOOK_URL=http://localhost:8080/webhook/news

# Lightweight notification for thread-events service
WEBHOOK_URL_THREAD_EVENTS=http://localhost:8000/webhooks/article-vectorized

# Webhook authentication
WEBHOOK_SIGNATURE=your-shared-secret-key
WEBHOOK_TIMEOUT=60
WEBHOOK_MAX_RETRIES=3

# -----------------------------------------------------------------------------
# NewsAPI.org (Optional)
# -----------------------------------------------------------------------------
NEWSAPI_KEY=your-newsapi-key-here

# -----------------------------------------------------------------------------
# Application Settings
# -----------------------------------------------------------------------------
APP_NAME=news-crawler-ai
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Toggles
ENABLE_WEBHOOKS=1  # 0 to disable
ENABLE_GENAI=1     # 0 to disable

# Timeouts
NEWS_FETCH_TIMEOUT=20  # seconds per fetch

# -----------------------------------------------------------------------------
# Model Cache
# -----------------------------------------------------------------------------
TRANSFORMERS_CACHE=./models/transformers

# -----------------------------------------------------------------------------
# Selenium / Browser Automation (if using dynamic scrapers)
# -----------------------------------------------------------------------------
CHROME_BIN=/Applications/Google Chrome.app/Contents/MacOS/Google Chrome
CHROMEDRIVER_PATH=/usr/local/bin/chromedriver
CHROMEDRIVER_VERSION=latest
CHROME_USER_DATA_DIR=/tmp/chrome-user-data

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
STATE_FILE=crawler_state.json
```

### 4.2 Migrar configuración hardcodeada

**Antes** (en `ingest/classifier.py`):
```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
```

**Después**:
```python
from news_crawler.core.config import GENAI_CONFIG, MONGO_CONFIG

# Usar directamente
api_key = GENAI_CONFIG.api_key
mongo_uri = MONGO_CONFIG.uri
```

---

## 5. Separación de Responsabilidades

### 5.1 Repositorio de Logs

Crear `news_crawler/repositories/pipeline_logs_repository.py`:

```python
"""Repositorio para logs del pipeline."""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from bson import ObjectId

from news_crawler.db.mongo_client import get_db
from news_crawler.core.config import MONGO_CONFIG

logger = logging.getLogger(__name__)

class PipelineLogsRepository:
    """Gestiona logs estructurados del pipeline en MongoDB."""
    
    def __init__(self):
        self.collection = get_db()[MONGO_CONFIG.logs_collection]
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Crea índices necesarios."""
        self.collection.create_index("ts")
        self.collection.create_index("actor")
        self.collection.create_index("action")
        self.collection.create_index("status")
        self.collection.create_index([("article_id", 1), ("action", 1)])
    
    def log_event(
        self,
        action: str,
        actor: str,
        article_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Registra un evento del pipeline.
        
        Args:
            action: Acción realizada (ej: "scrape_article", "classify", "send_webhook")
            actor: Componente que realizó la acción (ej: "bbc_scraper", "genai_provider")
            article_id: ID del artículo relacionado (opcional)
            batch_id: ID del batch relacionado (opcional)
            status: Estado del evento (success, error, warning)
            details: Información adicional
            error_message: Mensaje de error si status=error
        
        Returns:
            ID del log insertado
        """
        log_entry = {
            "ts": datetime.utcnow(),
            "action": action,
            "actor": actor,
            "article_id": article_id,
            "batch_id": batch_id,
            "status": status,
            "details": details or {},
            "error_message": error_message
        }
        
        try:
            result = self.collection.insert_one(log_entry)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert log: {e}")
            # No lanzar excepción para evitar interrumpir el pipeline
            return ""
    
    def get_logs(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        status: Optional[str] = None,
        article_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Recupera logs con filtros opcionales.
        
        Returns:
            Lista de logs ordenados por timestamp descendente
        """
        query = {}
        
        if actor:
            query["actor"] = actor
        if action:
            query["action"] = action
        if status:
            query["status"] = status
        if article_id:
            query["article_id"] = article_id
        if batch_id:
            query["batch_id"] = batch_id
        
        cursor = self.collection.find(query).sort("ts", -1).limit(limit)
        return list(cursor)
```

### 5.2 Refactorizar `classifier.py`

**Antes**: `ingest/classifier.py` tenía ~500 líneas mezclando orquestación, clasificación, llamadas a API, etc.

**Después**: Separar en:

1. **`news_crawler/core/orchestrator.py`** → Orquestación
2. **`news_crawler/processors/classifier.py`** → Solo clasificación
3. **`news_crawler/processors/sentiment_analyzer.py`** → Solo sentiment
4. **`news_crawler/providers/genai_provider.py`** → Solo llamadas a Gemini

Ejemplo de `news_crawler/processors/classifier.py` refactorizado:

```python
"""Clasificador de sentimiento y tópicos."""
import logging
from typing import Dict, Any
import torch
from transformers import pipeline

from news_crawler.core.config import APP_CONFIG

logger = logging.getLogger(__name__)

# Cargar modelos una sola vez
CACHE_DIR = APP_CONFIG.transformers_cache
DEVICE = 0 if torch.cuda.is_available() else -1

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=DEVICE,
    cache_dir=CACHE_DIR
)

CANDIDATE_TOPICS = [
    "politics and government",
    "sports and athletics",
    "science and research",
    # ... resto de tópicos
]

topic_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=DEVICE,
    cache_dir=CACHE_DIR
)

def classify_article(text: str) -> Dict[str, Any]:
    """
    Clasifica un artículo en sentiment y topic.
    
    Args:
        text: Texto del artículo
    
    Returns:
        Dict con sentiment y topic
    """
    if not text or len(text.strip()) < 20:
        return {
            "sentiment": {"label": "NEUTRAL", "score": 0.0},
            "topic": "other"
        }
    
    # Truncar si es muy largo
    text_sample = text[:512]
    
    result = {}
    
    # Sentiment
    try:
        sentiment_result = sentiment_pipeline(text_sample)[0]
        result["sentiment"] = {
            "label": sentiment_result["label"],
            "score": sentiment_result["score"]
        }
    except Exception as e:
        logger.error(f"Sentiment classification failed: {e}")
        result["sentiment"] = {"label": "NEUTRAL", "score": 0.0}
    
    # Topic
    try:
        topic_result = topic_pipeline(text_sample, candidate_labels=CANDIDATE_TOPICS)
        result["topic"] = topic_result["labels"][0]
    except Exception as e:
        logger.error(f"Topic classification failed: {e}")
        result["topic"] = "other"
    
    return result
```

---

## 6. Manejo de Estado y Persistencia

### 6.1 Integrar StateManager en el flujo

**En `news_crawler/core/orchestrator.py`:**

```python
# Al inicio del batch
self.state_manager.set_last_batch_id(batch_id)

# Al procesar cada scraper
scraper_name = article.get("source", "unknown")
self.state_manager.set_scraper_state(
    scraper_name=scraper_name,
    last_url=article.get("url"),
    articles_processed=1,  # incrementar
    last_run=datetime.utcnow().isoformat()
)

# Al finalizar
self.state_manager.increment_stats(stats["articles_processed"])
```

### 6.2 Agregar flag `--resume` al CLI

```python
# En CLI (ver sección 10)
if args.resume:
    last_batch = state_manager.get_last_batch_id()
    logger.info(f"Resuming from last batch: {last_batch}")
    # Lógica para reanudar desde último punto
```

---

## 7. Logging y Manejo de Errores

### 7.1 Configurar logging estructurado

Crear `news_crawler/utils/logging.py`:

```python
"""Configuración de logging estructurado."""
import logging
import sys
from typing import Optional

def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configura el sistema de logging.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Formato personalizado
        log_file: Archivo opcional para guardar logs
    """
    if format_string is None:
        format_string = (
            "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
        )
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Silenciar logs verbosos de librerías
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
```

**Uso en `main.py`:**

```python
from news_crawler.utils.logging import setup_logging
from news_crawler.core.config import APP_CONFIG

setup_logging(level=APP_CONFIG.log_level)
```

### 7.2 Logging con contexto

**Reemplazar:**
```python
print(f"Processing article: {title}")
```

**Con:**
```python
logger.info(
    f"Processing article: {title}",
    extra={
        "article_id": article_id,
        "source": source,
        "batch_id": batch_id
    }
)
```

### 7.3 Manejo de errores consistente

**Patrón recomendado:**

```python
def process_article(article: Dict[str, Any]) -> Optional[str]:
    """Procesa un artículo."""
    try:
        # Lógica principal
        result = do_something(article)
        logger.info(f"Article processed: {article['url']}")
        return result
        
    except ValueError as e:
        # Errores esperados (validación, etc.)
        logger.warning(f"Validation error for {article['url']}: {e}")
        return None
        
    except Exception as e:
        # Errores inesperados
        logger.error(
            f"Unexpected error processing {article['url']}: {e}",
            exc_info=True  # Incluye stack trace
        )
        # Registrar en MongoDB
        logs_repo.log_event(
            action="process_article",
            actor="orchestrator",
            article_id=article.get("id"),
            status="error",
            error_message=str(e)
        )
        return None
```

---

## 8. Gestión de Dependencias

### 8.1 Crear `pyproject.toml`

```toml
[project]
name = "news-crawler-ai"
version = "0.1.0"
description = "Automated news scraping, classification and enrichment pipeline"
authors = [
    {name = "Christian Fita", email = "your-email@example.com"}
]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    # Core scraping
    "httpx>=0.24.0",
    "trafilatura~=2.0.0",
    "beautifulsoup4~=4.13.5",
    "lxml>=4.9.0",
    "lxml-html-clean>=0.1.0",
    "feedparser~=6.0.12",
    
    # CLI & logging
    "typer>=0.9.0",
    "rich>=13.0.0",
    "tqdm>=4.65.0",
    "python-dotenv~=1.1.1",
    
    # Databases
    "pymongo~=4.15.0",
    
    # Numeric stack
    "numpy>=1.23,<2.0",
    
    # NLP / ML
    "transformers>=4.30,<4.37",
    "torch==2.2.2",
    "safetensors==0.4.3",
    
    # Web automation
    "selenium==4.27.1",
    "webdriver-manager==4.0.2",
    "urllib3[socks]==2.4.0",
    
    # APIs
    "requests~=2.32.5",
    "google-genai>=1.56.0",
    
    # Utilities
    "typing-extensions==4.12.2",
    "websocket-client==1.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0",
    "ipython>=8.0.0",
]

[project.scripts]
news-crawler = "news_crawler.cli:main"
news-crawler-bootstrap = "scripts.bootstrap_models:main"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=news_crawler --cov-report=html --cov-report=term"

[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.venv
  | build
  | dist
)/
'''

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "F", "W", "I", "N"]
ignore = ["E501"]  # line too long (manejado por black)

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # relajar para migración gradual
ignore_missing_imports = true
```

### 8.2 Migrar a Poetry

```bash
# Instalar Poetry si no está instalado
curl -sSL https://install.python-poetry.org | python3 -

# En el directorio del proyecto
poetry install

# Activar entorno
poetry shell

# Agregar nueva dependencia
poetry add nombre-paquete

# Agregar dependencia de desarrollo
poetry add --group dev pytest

# Actualizar dependencias
poetry update

# Exportar requirements.txt (para compatibilidad)
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

---

## 9. Testing y Calidad de Código

### 9.1 Crear estructura de tests

```
tests/
├── __init__.py
├── conftest.py                  # Fixtures compartidos
├── test_scrapers.py
├── test_processors.py
├── test_repositories.py
├── test_providers.py
└── test_orchestrator.py
```

#### `tests/conftest.py`

```python
"""Fixtures compartidos para tests."""
import pytest
from unittest.mock import Mock
from datetime import datetime

@pytest.fixture
def sample_article():
    """Artículo de ejemplo para tests."""
    return {
        "title": "Test Article Title",
        "url": "https://example.com/article",
        "text": "This is a test article with enough text to be valid. " * 10,
        "source": "test-source",
        "scraped_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def mock_mongo_client(monkeypatch):
    """Mock del cliente MongoDB."""
    mock_db = Mock()
    mock_collection = Mock()
    mock_db.__getitem__.return_value = mock_collection
    
    def mock_get_db():
        return mock_db
    
    monkeypatch.setattr("news_crawler.db.mongo_client.get_db", mock_get_db)
    return mock_collection

@pytest.fixture
def mock_genai_provider(monkeypatch):
    """Mock del proveedor GenAI."""
    mock_provider = Mock()
    mock_provider.clean_and_extract_entities.return_value = {
        "cleaned_text": "Cleaned test text",
        "locations": ["London"],
        "organizations": ["BBC"],
        "persons": ["Smith"]
    }
    return mock_provider
```

#### `tests/test_processors.py`

```python
"""Tests para procesadores de texto."""
import pytest
from news_crawler.processors.classifier import classify_article

def test_classify_article_success(sample_article):
    """Test clasificación exitosa."""
    result = classify_article(sample_article["text"])
    
    assert "sentiment" in result
    assert "topic" in result
    assert result["sentiment"]["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert isinstance(result["sentiment"]["score"], float)
    assert result["topic"] in [
        "politics and government",
        "sports and athletics",
        # ... resto de tópicos
        "other"
    ]

def test_classify_article_empty_text():
    """Test clasificación con texto vacío."""
    result = classify_article("")
    
    assert result["sentiment"]["label"] == "NEUTRAL"
    assert result["topic"] == "other"

def test_classify_article_short_text():
    """Test clasificación con texto muy corto."""
    result = classify_article("Too short")
    
    assert result["sentiment"]["label"] == "NEUTRAL"
    assert result["topic"] == "other"
```

#### `tests/test_repositories.py`

```python
"""Tests para repositorios."""
import pytest
from news_crawler.repositories.articles_repository import ArticlesRepository

def test_insert_article(mock_mongo_client, sample_article):
    """Test inserción de artículo."""
    repo = ArticlesRepository()
    
    # Configurar mock
    mock_mongo_client.insert_one.return_value.inserted_id = "123abc"
    
    # Ejecutar
    article_id = repo.insert_article(sample_article)
    
    # Verificar
    assert article_id == "123abc"
    mock_mongo_client.insert_one.assert_called_once()

def test_get_article_by_id(mock_mongo_client):
    """Test recuperación de artículo por ID."""
    repo = ArticlesRepository()
    
    # Configurar mock
    mock_mongo_client.find_one.return_value = {
        "_id": "123abc",
        "title": "Test Article"
    }
    
    # Ejecutar
    article = repo.get_article_by_id("123abc")
    
    # Verificar
    assert article is not None
    assert article["title"] == "Test Article"
```

### 9.2 Ejecutar tests

```bash
# Todos los tests
poetry run pytest

# Con coverage
poetry run pytest --cov=news_crawler --cov-report=html

# Tests específicos
poetry run pytest tests/test_processors.py

# Con verbosidad
poetry run pytest -v

# Ver coverage en navegador
open htmlcov/index.html
```

### 9.3 Configurar linters

```bash
# Formatear código con black
poetry run black news_crawler/ tests/

# Linting con ruff
poetry run ruff check news_crawler/ tests/

# Type checking con mypy
poetry run mypy news_crawler/

# Todo junto
poetry run black news_crawler/ tests/ && \
poetry run ruff check news_crawler/ tests/ && \
poetry run mypy news_crawler/ && \
poetry run pytest
```

---

## 10. CLI y Puntos de Entrada

### 10.1 Crear `news_crawler/cli.py`

```python
"""CLI principal de News-Crawler-AI."""
import sys
import logging
from typing import Optional
import typer
from rich.console import Console
from rich.logging import RichHandler

from news_crawler.core.config import APP_CONFIG
from news_crawler.core.state import StateManager
from news_crawler.core.orchestrator import PipelineOrchestrator
from news_crawler.utils.logging import setup_logging

app = typer.Typer(
    name="news-crawler",
    help="News Crawler AI - Automated news scraping and classification"
)
console = Console()

@app.command()
def run(
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-l",
        help="Maximum number of articles to process"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run without writing to database"
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from last state"
    ),
    reset_state: bool = typer.Option(
        False,
        "--reset-state",
        help="Reset state before running"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging"
    ),
):
    """
    Run the complete scraping and classification pipeline.
    """
    # Setup logging
    log_level = "DEBUG" if verbose else APP_CONFIG.log_level
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    console.print(f"[bold blue]News-Crawler-AI Pipeline[/bold blue]")
    console.print(f"Dry run: {dry_run}")
    console.print(f"Limit: {limit or 'unlimited'}")
    console.print()
    
    try:
        # State management
        state_manager = StateManager()
        
        if reset_state:
            state_manager.reset()
            console.print("[yellow]State reset[/yellow]")
        
        if resume:
            last_batch = state_manager.get_last_batch_id()
            if last_batch:
                console.print(f"[green]Resuming from batch: {last_batch}[/green]")
            else:
                console.print("[yellow]No previous state found[/yellow]")
        
        # Run pipeline
        orchestrator = PipelineOrchestrator(
            state_manager=state_manager,
            dry_run=dry_run
        )
        
        stats = orchestrator.run(limit=limit)
        
        # Display results
        console.print()
        console.print("[bold green]Pipeline completed successfully![/bold green]")
        console.print(f"Articles processed: {stats['articles_processed']}")
        console.print(f"Articles skipped: {stats['articles_skipped']}")
        console.print(f"Articles failed: {stats['articles_failed']}")
        console.print(f"Duration: {stats.get('duration_seconds', 0):.2f}s")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        console.print(f"\n[bold red]Pipeline failed: {e}[/bold red]")
        sys.exit(1)

@app.command()
def status():
    """Show pipeline status and statistics."""
    state_manager = StateManager()
    
    console.print("[bold]Pipeline Status[/bold]")
    console.print()
    
    # Last batch
    last_batch = state_manager.get_last_batch_id()
    if last_batch:
        console.print(f"Last batch ID: {last_batch}")
    else:
        console.print("No batches run yet")
    
    # Stats
    stats = state_manager.get_pipeline_stats()
    console.print(f"Total articles: {stats['total_articles_processed']}")
    console.print(f"Total batches: {stats['total_batches']}")
    if stats['last_run']:
        console.print(f"Last run: {stats['last_run']}")

@app.command()
def reset():
    """Reset pipeline state."""
    state_manager = StateManager()
    
    confirmed = typer.confirm("Are you sure you want to reset all state?")
    if confirmed:
        state_manager.reset()
        console.print("[green]State reset successfully[/green]")
    else:
        console.print("[yellow]Cancelled[/yellow]")

def main():
    """Entry point for CLI."""
    app()

if __name__ == "__main__":
    main()
```

### 10.2 Registrar entry points en `pyproject.toml`

```toml
[project.scripts]
news-crawler = "news_crawler.cli:main"
news-crawler-bootstrap = "scripts.bootstrap_models:main"
```

### 10.3 Uso del CLI

```bash
# Ejecutar pipeline completo
poetry run news-crawler run

# Dry run (sin escribir a BD)
poetry run news-crawler run --dry-run

# Limitar a 50 artículos
poetry run news-crawler run --limit 50

# Modo verbose
poetry run news-crawler run --verbose

# Reanudar desde último punto
poetry run news-crawler run --resume

# Reset state y ejecutar
poetry run news-crawler run --reset-state

# Ver estado
poetry run news-crawler status

# Reset manual
poetry run news-crawler reset
```

---

## 11. Documentación

### 11.1 Mejorar README.md

Seguir estructura de `thread_events_agents/README.md`:

**Secciones requeridas:**

1. ✅ Descripción general y propósito
2. ✅ Arquitectura (agregar diagrama de flujo ASCII)
3. ✅ Tech stack
4. ✅ Instalación paso a paso
5. ✅ Configuración (variables de entorno)
6. ✅ Uso (comandos CLI)
7. ✅ Estructura de proyecto
8. ✅ Modelos de datos (MongoDB collections)
9. ⭐ **NUEVO**: Flujo de procesamiento detallado
10. ⭐ **NUEVO**: Decisiones arquitectónicas
11. ⭐ **NUEVO**: Troubleshooting común
12. ✅ Desarrollo y contribución

### 11.2 Agregar docstrings

**Todos los módulos públicos deben tener docstrings:**

```python
"""
Módulo para scraping de BBC News.

Este módulo implementa un scraper basado en RSS feeds que:
- Extrae artículos de múltiples feeds de BBC
- Valida la longitud mínima del texto
- Deduplica por URL

Example:
    >>> from news_crawler.scrapers.bbc_scraper import BBCScraper
    >>> scraper = BBCScraper()
    >>> for article in scraper.scrape():
    ...     print(article['title'])
"""
```

**Todas las funciones públicas:**

```python
def classify_article(text: str) -> Dict[str, Any]:
    """
    Clasifica un artículo en sentiment y topic usando transformers.
    
    Esta función utiliza dos pipelines:
    - DistilBERT para análisis de sentimiento (POSITIVE/NEGATIVE/NEUTRAL)
    - BART-MNLI para clasificación zero-shot en 12 tópicos predefinidos
    
    Args:
        text: Texto completo del artículo a clasificar
    
    Returns:
        Dict con dos keys:
            - sentiment: Dict con label (str) y score (float)
            - topic: String con la categoría asignada
    
    Raises:
        ValueError: Si el texto está vacío o es None
    
    Example:
        >>> result = classify_article("Stock markets rose 3% today...")
        >>> print(result)
        {
            'sentiment': {'label': 'POSITIVE', 'score': 0.92},
            'topic': 'business and finance'
        }
    
    Note:
        Textos muy cortos (<20 chars) retornan clasificación neutral por defecto.
    """
```

### 11.3 Crear diagramas de flujo

Agregar a README.md:

```markdown
## 🔄 Flujo de Procesamiento Detallado

```
┌──────────────────────────────────────────────────────────────────┐
│                     INICIO: CLI Command                           │
│          poetry run news-crawler run --limit 100                  │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  PipelineOrchestrator│
        │  - Load state        │
        │  - Generate batch_id │
        └──────────┬───────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
    ┌─────────────┐    ┌──────────────┐
    │ StateManager│    │ Repositories │
    │ (resume)    │    │ (MongoDB)    │
    └──────┬──────┘    └──────┬───────┘
           │                  │
           └──────────┬───────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │ get_all_articles()              │
        │ - Iterate all scrapers          │
        │ - Check link_pool for dupes     │
        └─────────────┬───────────────────┘
                      │
              ┌───────┴──────────┐
              │                  │
              ▼                  ▼
    ┌──────────────────┐  ┌─────────────────────┐
    │ BBCScraper       │  │ CNNScraper          │
    │ - Parse RSS      │  │ - Parse RSS         │
    │ - Extract text   │  │ - Extract text      │
    └──────────┬───────┘  └─────────┬───────────┘
               │                    │
               └────────┬───────────┘
                        │
                        ▼
        ┌──────────────────────────────────┐
        │ Para cada artículo:              │
        │ 1. Check duplicate in link_pool  │
        └─────────────┬────────────────────┘
                      │
              ┌───────┴──────────┐
              │ Already          │
              │ processed?       │
              └────┬──────────┬──┘
                   │Yes       │No
                   ▼          ▼
              ┌────────┐  ┌─────────────────────┐
              │ Skip   │  │ GenAIProvider        │
              └────────┘  │ - Clean text         │
                          │ - Extract entities   │
                          └─────────┬────────────┘
                                    │
                                    ▼
                          ┌──────────────────────┐
                          │ Summarizer           │
                          │ - BART if >200 tok   │
                          └─────────┬────────────┘
                                    │
                                    ▼
                          ┌──────────────────────┐
                          │ Classifier           │
                          │ - Sentiment          │
                          │ - Topic              │
                          └─────────┬────────────┘
                                    │
                            ┌───────┴─────────┐
                            │                 │
                            ▼                 ▼
                  ┌──────────────────┐  ┌────────────┐
                  │ ArticlesRepo     │  │ LinkPoolRepo│
                  │ - insert_article │  │ - mark_as  │
                  │                  │  │   _processed│
                  └──────────┬───────┘  └────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ WebhookProvider   │
                  │ - send to         │
                  │   embedding svc   │
                  │ - send to         │
                  │   thread-events   │
                  └──────────┬────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │ PipelineLogsRepo │
                  │ - log_event      │
                  └──────────┬────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ MetadataRepo         │
                  │ - Save batch stats   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ StateManager         │
                  │ - update stats       │
                  │ - save state         │
                  └──────────────────────┘
                             │
                             ▼
                  ┌──────────────────────────┐
                  │ FIN: Return stats        │
                  └──────────────────────────┘
```
```

