import os
import time
import json
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
import cohere
from groq import Groq
import faiss
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

# =============================================================================
# ENHANCED CONFIGURATION AND VALIDATION
# =============================================================================

@dataclass
class APIConfig:
    """Enhanced API configuration with validation"""
    news_api_key: str
    alpha_vantage_key: str
    cohere_api_key: str
    groq_api_key: str
    base_news_url: str = "https://newsapi.org/v2"
    base_av_url: str = "https://www.alphavantage.co/query"

class APIValidator:
    """Validates API keys before analysis"""
    
    @staticmethod
    def validate_news_api(api_key: str) -> Tuple[bool, str]:
        """Test NewsAPI key with minimal request"""
        try:
            response = requests.get(
                f"https://newsapi.org/v2/top-headlines?country=us&pageSize=1&apiKey={api_key}",
                timeout=10
            )
            if response.status_code == 200:
                return True, "✅ NewsAPI key is valid"
            elif response.status_code == 401:
                return False, "❌ NewsAPI key is invalid"
            elif response.status_code == 429:
                return False, "⏰ NewsAPI rate limit exceeded"
            else:
                return False, f"❌ NewsAPI error: {response.status_code}"
        except Exception as e:
            return False, f"❌ NewsAPI connection failed: {str(e)}"
    
    @staticmethod
    def validate_alpha_vantage(api_key: str) -> Tuple[bool, str]:
        """Test Alpha Vantage key with minimal request"""
        try:
            response = requests.get(
                f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={api_key}&outputsize=compact",
                timeout=15
            )
            if response.status_code == 200:
                data = response.json()
                if "Error Message" in data:
                    return False, "❌ Alpha Vantage: Invalid symbol or API issue"
                elif "Note" in data and "call frequency" in data["Note"]:
                    return False, "⏰ Alpha Vantage rate limit exceeded"
                elif "Time Series" in str(data) or "Meta Data" in data:
                    return True, "✅ Alpha Vantage key is valid"
                else:
                    return False, "❌ Alpha Vantage key might be invalid"
            else:
                return False, f"❌ Alpha Vantage error: {response.status_code}"
        except Exception as e:
            return False, f"❌ Alpha Vantage connection failed: {str(e)}"
    
    @staticmethod
    def validate_cohere(api_key: str) -> Tuple[bool, str]:
        """Test Cohere key with minimal request"""
        try:
            client = cohere.Client(api_key)
            response = client.embed(
                texts=["test"],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return True, "✅ Cohere key is valid"
        except Exception as e:
            if "invalid api token" in str(e).lower():
                return False, "❌ Cohere API key is invalid"
            elif "rate limit" in str(e).lower():
                return False, "⏰ Cohere rate limit exceeded"
            else:
                return False, f"❌ Cohere error: {str(e)}"
    
    @staticmethod
    def validate_groq(api_key: str) -> Tuple[bool, str]:
        """Test Groq key with minimal request"""
        try:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="llama3-8b-8192",
                max_tokens=5
            )
            return True, "✅ Groq key is valid"
        except Exception as e:
            if "invalid api key" in str(e).lower():
                return False, "❌ Groq API key is invalid"
            elif "rate limit" in str(e).lower():
                return False, "⏰ Groq rate limit exceeded"
            else:
                return False, f"❌ Groq error: {str(e)}"

class RateLimiter:
    """Manages API rate limits across different services"""
    
    def __init__(self):
        if 'rate_limiter' not in st.session_state:
            st.session_state.rate_limiter = {
                'news': [],
                'alpha_vantage': [],
                'cohere': [],
                'groq': []
            }
    
    def can_make_call(self, api_type: str, limit_per_minute: int = 5) -> bool:
        """Check if we can make an API call within rate limits"""
        now = time.time()
        calls = st.session_state.rate_limiter.get(api_type, [])
        recent_calls = [t for t in calls if now - t < 60]
        st.session_state.rate_limiter[api_type] = recent_calls
        return len(recent_calls) < limit_per_minute
    
    def record_call(self, api_type: str):
        """Record an API call timestamp"""
        if api_type in st.session_state.rate_limiter:
            st.session_state.rate_limiter[api_type].append(time.time())

# =============================================================================
# ENHANCED DATA CLIENTS WITH BETTER ERROR HANDLING
# =============================================================================

class EnhancedNewsAPIClient:
    """Enhanced News API client with better error handling and rate limiting"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.rate_limiter = RateLimiter()
        
        self.country_codes = {
            'Global': None,
            'United States': 'us', 'United Kingdom': 'gb', 'Canada': 'ca',
            'Australia': 'au', 'Germany': 'de', 'France': 'fr', 'Italy': 'it',
            'Spain': 'es', 'Netherlands': 'nl', 'Belgium': 'be', 'Switzerland': 'ch',
            'Sweden': 'se', 'Norway': 'no', 'Denmark': 'dk', 'Finland': 'fi',
            'Japan': 'jp', 'South Korea': 'kr', 'China': 'cn', 'India': 'in',
            'Singapore': 'sg', 'Hong Kong': 'hk', 'Brazil': 'br', 'Mexico': 'mx',
            'Argentina': 'ar', 'South Africa': 'za', 'Israel': 'il', 'Turkey': 'tr',
            'Russia': 'ru', 'UAE': 'ae', 'Saudi Arabia': 'sa', 'Egypt': 'eg',
            'Nigeria': 'ng', 'Kenya': 'ke', 'Thailand': 'th', 'Vietnam': 'vn',
            'Philippines': 'ph', 'Indonesia': 'id', 'Malaysia': 'my'
        }
        
        self.categories = [
            'general', 'business', 'entertainment', 'health', 
            'science', 'sports', 'technology'
        ]
    
    def safe_api_call(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make safe API call with enhanced error handling"""
        if not self.rate_limiter.can_make_call('news', 60):  # NewsAPI: 1000/day
            st.warning("⏰ NewsAPI rate limit reached. Please wait before making another request.")
            return None
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            self.rate_limiter.record_call('news')
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    return data
                else:
                    st.error(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
                    return None
            elif response.status_code == 401:
                st.error("🔑 NewsAPI: Invalid API key. Please check your configuration.")
                return None
            elif response.status_code == 429:
                st.warning("⏰ NewsAPI: Rate limit exceeded. Please try again later.")
                return None
            else:
                st.error(f"NewsAPI HTTP Error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("⏱️ NewsAPI request timed out. Please try again.")
            return None
        except requests.exceptions.ConnectionError:
            st.error("🌐 Cannot connect to NewsAPI. Check your internet connection.")
            return None
        except Exception as e:
            st.error(f"Unexpected NewsAPI error: {str(e)}")
            return None
    
    def get_custom_news(self, 
                       topic: str, 
                       country: str = None, 
                       category: str = None,
                       from_date: str = None, 
                       to_date: str = None,
                       language: str = 'en',
                       sort_by: str = 'relevancy', 
                       page_size: int = 50) -> List[Dict]:
        """Get news with enhanced error handling"""
        
        params = {
            'apiKey': self.api_key,
            'language': language,
            'sortBy': sort_by,
            'pageSize': min(page_size, 100)
        }
        
        if topic or from_date or to_date:
            endpoint = f"{self.base_url}/everything"
            if topic:
                params['q'] = topic
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date
        else:
            endpoint = f"{self.base_url}/top-headlines"
            if country and country in self.country_codes and self.country_codes[country]:
                params['country'] = self.country_codes[country]
            if category and category in self.categories:
                params['category'] = category
        
        data = self.safe_api_call(endpoint, params)
        if data:
            articles = data.get('articles', [])
            return [article for article in articles 
                   if article.get('title') and article.get('description')]
        return []

class EnhancedAlphaVantageClient:
    """Enhanced Alpha Vantage client with better error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limiter = RateLimiter()
        
        self.global_indices = {
            'US': ['SPY', 'QQQ', 'DIA', 'IWM'],
            'Europe': ['EWG', 'EWU', 'EWQ', 'EWI'],
            'Asia': ['EWJ', 'EWY', 'INDA', 'EWS'],
            'Emerging': ['EEM', 'VWO', 'EWZ', 'RSX']
        }
        
        self.sector_etfs = {
            'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial': 'XLF',
            'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP',
            'Industrial': 'XLI', 'Materials': 'XLB', 'Real Estate': 'XLRE',
            'Utilities': 'XLU', 'Communication': 'XLC'
        }
    
    def safe_api_call(self, params: Dict) -> Optional[Dict]:
        """Make safe API call with enhanced error handling"""
        if not self.rate_limiter.can_make_call('alpha_vantage', 5):  # 5 calls per minute
            st.warning("⏰ Alpha Vantage rate limit reached. Please wait before making another request.")
            return None
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            self.rate_limiter.record_call('alpha_vantage')
            
            if response.status_code == 200:
                data = response.json()
                if "Error Message" in data:
                    st.warning(f"Alpha Vantage Error: {data['Error Message']}")
                    return None
                elif "Note" in data:
                    st.warning(f"Alpha Vantage Rate Limit: {data['Note']}")
                    return None
                return data
            else:
                st.error(f"Alpha Vantage HTTP Error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Alpha Vantage request timed out. Please try again.")
            return None
        except requests.exceptions.ConnectionError:
            st.error("🌐 Cannot connect to Alpha Vantage. Check your internet connection.")
            return None
        except Exception as e:
            st.error(f"Unexpected Alpha Vantage error: {str(e)}")
            return None
    
    def get_custom_market_data(self, 
                              symbols: List[str], 
                              interval: str = "daily",
                              region: str = "US") -> Dict[str, Dict]:
        """Get market data with enhanced error handling"""
        results = {}
        function_map = {
            "daily": "TIME_SERIES_DAILY",
            "weekly": "TIME_SERIES_WEEKLY",  
            "monthly": "TIME_SERIES_MONTHLY",
            "intraday": "TIME_SERIES_INTRADAY"
        }
        
        for symbol in symbols[:10]:  # Limit to 10 symbols to avoid rate limits
            params = {
                'function': function_map.get(interval, "TIME_SERIES_DAILY"),
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            if interval == "intraday":
                params['interval'] = '60min'
            
            data = self.safe_api_call(params)
            if data:
                results[symbol] = data
            
            # Small delay to respect rate limits
            time.sleep(0.2)
        
        return results

# =============================================================================
# ENHANCED DATA PROCESSOR WITH CACHING
# =============================================================================

class EnhancedFinancialDataProcessor:
    """Enhanced data processor with caching and better error handling"""
    
    def __init__(self, cohere_api_key: str):
        self.cohere_client = cohere.Client(cohere_api_key)
        self.rate_limiter = RateLimiter()
        self.index = None
        self.documents = []
    
    def process_custom_news_data(self, articles: List[Dict], topic: str, country: str = None) -> List[Dict]:
        """Process news data with enhanced structure"""
        processed = []
        for article in articles:
            if article.get('title') and article.get('description'):
                content_parts = [
                    f"Topic: {topic}",
                    f"Title: {article['title']}",
                    f"Description: {article['description']}"
                ]
                if country and country != 'Global':
                    content_parts.insert(1, f"Country: {country}")
                if article.get('content'):
                    content_parts.append(f"Content: {article['content'][:500]}...")
                
                processed.append({
                    'title': article['title'],
                    'description': article['description'],
                    'content': " | ".join(content_parts),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'topic': topic,
                    'country': country,
                    'type': 'custom_news'
                })
        return processed
    
    def process_enhanced_market_data(self, market_data: Dict[str, Dict], context: Dict) -> List[Dict]:
        """Process market data with enhanced structure"""
        processed = []
        for symbol, data in market_data.items():
            if not data:
                continue
                
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key or 'FX' in key:
                    time_series_key = key
                    break
            
            if not time_series_key or time_series_key not in data:
                continue
            
            time_series = data[time_series_key]
            for date, values in list(time_series.items())[:30]:
                try:
                    if '1. open' in values:
                        content = f"Symbol: {symbol} | Date: {date} | Region: {context.get('region', 'Global')} | Open: ${values['1. open']} | High: ${values['2. high']} | Low: ${values['3. low']} | Close: ${values['4. close']} | Volume: {values['5. volume']}"
                        processed.append({
                            'symbol': symbol,
                            'date': date,
                            'content': content,
                            'open': float(values['1. open']),
                            'high': float(values['2. high']),
                            'low': float(values['3. low']),
                            'close': float(values['4. close']),
                            'volume': int(values['5. volume']),
                            'region': context.get('region', 'Global'),
                            'type': 'enhanced_market_data'
                        })
                    elif 'close' in values:
                        content = f"Symbol: {symbol} | Date: {date} | Region: {context.get('region', 'Global')} | Close: ${values['close']}"
                        processed.append({
                            'symbol': symbol,
                            'date': date,
                            'content': content,
                            'close': float(values['close']),
                            'region': context.get('region', 'Global'),
                            'type': 'enhanced_market_data'
                        })
                except (ValueError, KeyError):
                    continue
        return processed
    
    def build_enhanced_vector_index(self, documents: List[Dict]):
        """Build vector index with enhanced error handling"""
        if not documents:
            st.warning("No documents to index")
            return
        
        self.documents = documents
        texts = [doc['content'] for doc in documents if doc.get('content')]
        
        if not texts:
            st.warning("No valid content found in documents")
            return
        
        if not self.rate_limiter.can_make_call('cohere', 10):  # Cohere rate limit
            st.warning("⏰ Cohere rate limit reached. Please wait before indexing.")
            return
        
        try:
            batch_size = 96
            all_embeddings = []
            
            with st.spinner(f"🔍 Generating embeddings for {len(texts)} documents..."):
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    response = self.cohere_client.embed(
                        texts=batch_texts,
                        model="embed-english-v3.0",
                        input_type="search_document"
                    )
                    all_embeddings.extend(response.embeddings)
                    self.rate_limiter.record_call('cohere')
                    
                    # Small delay between batches
                    if i + batch_size < len(texts):
                        time.sleep(1)
            
            embeddings = np.array(all_embeddings)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            st.success(f"✅ Successfully indexed {len(texts)} documents")
            
        except Exception as e:
            st.error(f"❌ Error generating embeddings: {str(e)}")
            return
    
    def search_enhanced_similar(self, query: str, k: int = 5, filters: Dict = None) -> List[Tuple[Dict, float]]:
        """Search similar documents with enhanced error handling"""
        if not self.index or not self.documents:
            return []
        
        if not self.rate_limiter.can_make_call('cohere', 20):
            st.warning("⏰ Cohere search rate limit reached. Please wait.")
            return []
        
        try:
            response = self.cohere_client.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            self.rate_limiter.record_call('cohere')
            
            query_embedding = np.array(response.embeddings)
            search_k = min(k * 3, len(self.documents))
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    if filters:
                        skip = False
                        for filter_key, filter_value in filters.items():
                            if filter_key in doc and doc[filter_key] != filter_value:
                                skip = True
                                break
                        if skip:
                            continue
                    results.append((doc, distances[0][i]))
                    if len(results) >= k:
                        break
            return results
        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")
            return []

# =============================================================================
# ENHANCED AI ADVISOR WITH BETTER CONTEXT
# =============================================================================

class EnhancedFinancialAdvisor:
    """Enhanced financial advisor with improved context and error handling"""
    
    def __init__(self, groq_api_key: str, cohere_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.data_processor = EnhancedFinancialDataProcessor(cohere_api_key)
        self.rate_limiter = RateLimiter()
        self.conversation_history = []
    
    def _safe_groq_call(self, prompt: str, context: Dict, model: str = "llama3-8b-8192") -> str:
        """Make safe Groq API call with enhanced error handling"""
        if not self.rate_limiter.can_make_call('groq', 10):
            return "⏰ AI service rate limit reached. Please wait before asking another question."
        
        try:
            system_message = f"""You are FinWise AI, a sarcastic and brutally honest financial teacher who doesn't sugarcoat things. You're here to educate users like they're in a tough finance class, throwing in sharp wit and zero patience for dumb questions. Your job is to:

1. Teach financial concepts with clear explanations, analogies, and examples.
2. Answer questions directly, but don't hold back on pointing out obvious mistakes or gaps in the user's thinking.
3. Use the latest analysis report to ground your answers when relevant (matches the topic and is fresh within 24 hours). If it's not, tell the user why their request is pointless.
4. Use recent conversation history to stay on track, but don't bore them with old stuff.
5. Only give investment advice if the analysis report is relevant. If not, explain why.
6. Keep responses under 600 words, structured with headings for clarity.

Current Analysis Context:
- Topic: {context.get('topic', 'General Finance')}
- Geographic Focus: {context.get('country', 'Global')}
- Time Period: {context.get('date_range', 'Recent')}
- Data Sources: {context.get('data_sources', 'News and Market Data')}

Analysis Report Summary:
{self._get_report_summary(context.get('topic', ''))}

Conversation History (Last 3 Exchanges):
{self._get_conversation_summary(context.get('topic', ''))}

Be rude, but professional. Call out nonsense, use sarcasm, and make it educational. Always end with a disclaimer that this isn't personalized financial advice."""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.5,
                max_tokens=1500
            )
            self.rate_limiter.record_call('groq')
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            if "rate limit" in str(e).lower():
                return "⏰ AI service is currently overloaded. Please try again in a moment."
            elif "invalid api key" in str(e).lower():
                return "🔑 AI service authentication failed. Please check your API key configuration."
            else:
                return f"❌ AI service error: {str(e)}"
    
    def _get_conversation_summary(self, topic: str) -> str:
        """Summarize last 3 relevant conversations for context"""
        if not self.conversation_history:
            return "No prior conversations. You're starting from scratch, genius."
        
        relevant_history = [
            chat for chat in self.conversation_history[-3:]
            if topic.lower() in chat['query'].lower() or (chat['filters'] and topic.lower() in str(chat['filters']).lower())
        ]
        
        if not relevant_history:
            return "No relevant chats recently. Try asking something that makes sense with your analysis."
        
        summary = []
        for i, chat in enumerate(relevant_history, 1):
            summary.append(f"Chat {i}: You asked: '{chat['query'][:50]}{'...' if len(chat['query']) > 50 else ''}'. I said: '{chat['response'][:100]}{'...' if len(chat['response']) > 100 else ''}'")
        return "\n".join(summary)
    
    def _get_report_summary(self, topic: str) -> str:
        """Summarize the latest analysis report if relevant"""
        if 'analysis_results' not in st.session_state or 'processed_data' not in st.session_state:
            return "No analysis report available. Run an analysis first, genius."
        
        analysis = st.session_state.analysis_results
        processed_data = st.session_state.processed_data
        analysis_topic = analysis['metadata'].get('topic', '').lower()
        fetch_time = datetime.fromisoformat(analysis['metadata'].get('fetch_timestamp', '1970-01-01'))
        is_recent = (datetime.now() - fetch_time).total_seconds() < 24 * 3600
        is_topic_match = topic.lower() in analysis_topic or analysis_topic in topic.lower()
        
        if not (is_recent and is_topic_match):
            return f"No relevant report. Current report on '{analysis_topic}' is {'outdated' if not is_recent else 'off-topic'}. Run a new analysis."
        
        summary = [
            f"Topic: {analysis['metadata']['topic']}",
            f"Region: {analysis['metadata'].get('country', 'Global')}",
            f"Period: {analysis['metadata']['date_range']}",
            f"News Articles: {analysis['news_count']}",
            f"Market Data Points: {analysis['market_count']}"
        ]
        
        if 'sentiment_analysis' in analysis:
            summary.append(f"Sentiment: {analysis['sentiment_analysis'][:100]}{'...' if len(analysis['sentiment_analysis']) > 100 else ''}")
        if 'market_analysis' in analysis:
            summary.append(f"Market Trends: {analysis['market_analysis'][:100]}{'...' if len(analysis['market_analysis']) > 100 else ''}")
        if processed_data.get('news_data'):
            top_headlines = [f"- {article['title'][:80]}{'...' if len(article['title']) > 80 else ''}" 
                            for article in processed_data['news_data'][:3]]
            summary.append("Top Headlines:\n" + "\n".join(top_headlines))
        
        return "\n".join(summary)
    
    def analyze_custom_topic_sentiment(self, news_data: List[Dict], topic: str, country: str = None) -> str:
        """Analyze sentiment with enhanced context"""
        if not news_data:
            return "No news data available. Did you even run an analysis, or are you just wasting my time?"
        
        news_content = "\n".join([
            f"- {article['title']}: {article['description'][:200]}..." 
            for article in news_data[:15]
        ])
        
        geographic_context = f" in {country}" if country and country != 'Global' else " globally"
        prompt = f"""
        Analyze the sentiment around "{topic}"{geographic_context} based on this news:
        
        {news_content}
        
        As a snarky financial teacher, do this:
        1. Give the overall sentiment (bullish/bearish/neutral) and explain it like I'm a rookie who barely gets it.
        2. List key themes and developments, with examples to hammer it into my head.
        3. Highlight {country if country and country != 'Global' else 'global'} factors, because apparently I need to know this.
        4. Spell out market implications for {topic}-related investments, since I can't figure it out myself.
        5. Point out risks and opportunities, because I'd probably miss them.
        6. Tell me what to watch next, so I don't screw this up.
        
        Keep it under 600 words. Educate me, don't coddle me.
        """
        
        context = {
            'topic': topic,
            'country': country,
            'data_sources': 'Recent News Articles'
        }
        
        return self._safe_groq_call(prompt, context)
    
    def generate_custom_market_analysis(self, market_data: Dict, context: Dict) -> str:
        """Generate market analysis with enhanced context"""
        if not market_data:
            return "No market data. Seriously, did you forget to add stock symbols or what?"
        
        market_summary = []
        for symbol, data in market_data.items():
            if data and isinstance(data, dict):
                time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
                if time_series_key and time_series_key in data:
                    time_series = data[time_series_key]
                    recent_dates = list(time_series.keys())[:5]
                    if recent_dates:
                        latest_data = time_series[recent_dates[0]]
                        oldest_data = time_series[recent_dates[-1]]
                        try:
                            latest_close = float(latest_data.get('4. close', latest_data.get('close', 0)))
                            oldest_close = float(oldest_data.get('4. close', oldest_data.get('close', 0)))
                            if oldest_close > 0:
                                change_pct = ((latest_close - oldest_close) / oldest_close) * 100
                                market_summary.append(f"{symbol}: ${latest_close:.2f} ({change_pct:+.2f}% over 5 days)")
                        except (ValueError, TypeError):
                            market_summary.append(f"{symbol}: Data available but incomplete")
        
        market_text = "\n".join(market_summary) if market_summary else "Market data processing incomplete"
        region = context.get('region', 'Global')
        topic = context.get('topic', 'Market Analysis')
        
        prompt = f"""
        Analyze this market data for {region} markets on {topic}:
        
        {market_text}
        
        You're a financial teacher with zero patience. Do this:
        1. Give a regional performance overview, like I don't know what a stock chart is.
        2. Explain {topic}-specific trends, with analogies so I get it.
        3. Compare instruments, because I'm too lazy to do it myself.
        4. Highlight {region} market factors, since I apparently need a geography lesson.
        5. Provide technical and fundamental insights, or I'll miss the obvious.
        6. List risks and opportunities, because I'd probably tank my portfolio.
        7. Suggest next steps for {region} exposure, if I'm not totally hopeless.
        
        Educate me in under 600 words. Don't waste my time.
        """
        
        return self._safe_groq_call(prompt, context)
    
    def answer_custom_financial_query(self, query: str, filters: Dict = None) -> str:
        """Answer financial queries with enhanced context"""
        similar_docs = self.data_processor.search_enhanced_similar(query, k=5, filters=filters)
        
        if similar_docs:
            context_docs = []
            for doc, score in similar_docs:
                relevance = "High" if score < 0.5 else "Medium" if score < 1.0 else "Low"
                context_docs.append(f"[{relevance} Relevance] {doc['content']}")
            context_text = "\n\n".join(context_docs)
            source_info = f"Based on {len(similar_docs)} relevant documents from recent data"
        else:
            context_text = "No specific recent data available for this query."
            source_info = "Using general financial knowledge and analysis report"
        
        # Check analysis relevance for advice and report usage
        advice_eligible = False
        advice_reason = ""
        report_summary = self._get_report_summary(filters.get('topic', query) if filters else query)
        
        if 'analysis_results' in st.session_state:
            analysis = st.session_state.analysis_results
            analysis_topic = analysis['metadata'].get('topic', '').lower()
            fetch_time = datetime.fromisoformat(analysis['metadata'].get('fetch_timestamp', '1970-01-01'))
            is_recent = (datetime.now() - fetch_time).total_seconds() < 24 * 3600
            is_topic_match = query.lower() in analysis_topic or analysis_topic in query.lower()
            
            if is_recent and is_topic_match and analysis.get('news_count', 0) > 0:
                advice_eligible = True
            else:
                advice_reason = f"Analysis is {'too old' if not is_recent else 'off-topic'} or lacks data. Run a relevant analysis first, genius."
        
        prompt = f"""
        Context from recent financial data:
        {context_text}
        
        Analysis Report Summary:
        {report_summary}
        
        User Question: {query}
        
        You're a rude financial teacher. Answer like I'm a student who didn't do the reading:
        1. **Direct Answer**: Answer the question bluntly. Call out any dumb assumptions.
        2. **Financial Lesson**: Explain a related financial concept with an analogy and example.
        3. **Report Insights**: Summarize key findings from the analysis report if relevant.
        4. **Evidence**: Use the report or similar data to back it up. Don't make stuff up.
        5. **Market Context**: Tie it to broader markets, since I clearly don't get it.
        6. **Risks**: List what could go wrong, because I'd probably ignore them.
        7. **Advice (if relevant)**: {'Give actionable advice based on the analysis report.' if advice_eligible else f'No advice because: {advice_reason}'}
        
        Data Source: {source_info}
        Keep it under 600 words. Educate me, don't bore me.
        """
        
        analysis_context = {
            'topic': filters.get('topic', 'Custom Query') if filters else 'Custom Query',
            'country': filters.get('country', 'Global') if filters else 'Global',
            'data_sources': source_info
        }
        
        response = self._safe_groq_call(prompt, analysis_context)
        
        self.conversation_history.append({
            'query': query,
            'response': response,
            'filters': filters,
            'timestamp': datetime.now().isoformat()
        })
        
        return response

# =============================================================================
# ENHANCED DATA PIPELINE
# =============================================================================

class EnhancedFinWiseDataPipeline:
    """Enhanced data pipeline with better error handling and user feedback"""
    
    def __init__(self, config: APIConfig):
        self.news_client = EnhancedNewsAPIClient(config.news_api_key)
        self.av_client = EnhancedAlphaVantageClient(config.alpha_vantage_key)
        self.data_processor = EnhancedFinancialDataProcessor(config.cohere_api_key)
        self.advisor = EnhancedFinancialAdvisor(config.groq_api_key, config.cohere_api_key)
    
    def estimate_api_usage(self, topic: str, symbols: List[str], date_range: str) -> Dict[str, int]:
        """Estimate API usage for user transparency"""
        return {
            'news_calls': min(100, len(topic.split()) * 10),  # Rough estimate
            'market_calls': len(symbols) if symbols else 0,
            'cohere_calls': 20,  # For embeddings
            'groq_calls': 3  # For analysis
        }
    
    def fetch_custom_data(self, 
                         topic: str,
                         country: str = None,
                         from_date: str = None,
                         to_date: str = None,
                         stock_symbols: List[str] = None,
                         region: str = "US") -> Dict:
        """Fetch custom data with enhanced progress tracking"""
        
        # Initialize results
        results = {
            'news_data': [],
            'market_data': {},
            'processed_news': [],
            'processed_market': [],
            'metadata': {
                'topic': topic,
                'country': country,
                'date_range': f"{from_date} to {to_date}" if from_date and to_date else "Recent",
                'region': region,
                'fetch_timestamp': datetime.now().isoformat()
            }
        }
        
        # Progress tracking
        total_steps = 2 + (1 if stock_symbols else 0)
        current_step = 0
        
        # Fetch news data
        progress_placeholder = st.empty()
        with progress_placeholder:
            st.info(f"📰 Fetching news for '{topic}' {f'in {country}' if country and country != 'Global' else 'globally'}...")
            
        news_data = self.news_client.get_custom_news(
            topic=topic,
            country=country if country != 'Global' else None,
            from_date=from_date,
            to_date=to_date
        )
        
        current_step += 1
        results['news_data'] = news_data
        
        with progress_placeholder:
            st.success(f"✅ Found {len(news_data)} news articles")
            time.sleep(0.5)
        
        # Process news data
        if news_data:
            with progress_placeholder:
                st.info("🔄 Processing news data...")
            
            results['processed_news'] = self.data_processor.process_custom_news_data(
                news_data, topic, country
            )
            current_step += 1
            
            with progress_placeholder:
                st.success(f"✅ Processed {len(results['processed_news'])} news articles")
                time.sleep(0.5)
        
        # Fetch market data if symbols provided
        if stock_symbols:
            with progress_placeholder:
                st.info(f"📈 Fetching market data for {len(stock_symbols)} symbols...")
            
            market_data = self.av_client.get_custom_market_data(
                stock_symbols, 
                region=region
            )
            results['market_data'] = market_data
            
            if market_data:
                market_context = {'region': region, 'topic': topic}
                results['processed_market'] = self.data_processor.process_enhanced_market_data(
                    market_data, market_context
                )
            
            current_step += 1
            
            with progress_placeholder:
                st.success(f"✅ Retrieved data for {len(market_data)} symbols")
                time.sleep(0.5)
        
        progress_placeholder.empty()
        return results
    
    def build_custom_analysis(self, custom_data: Dict) -> Dict:
        """Build analysis with enhanced progress tracking"""
        
        # Combine all documents
        all_documents = []
        if custom_data['processed_news']:
            all_documents.extend(custom_data['processed_news'])
        if custom_data['processed_market']:
            all_documents.extend(custom_data['processed_market'])
        
        # Build vector index
        if all_documents:
            self.data_processor.build_enhanced_vector_index(all_documents)
        
        # Initialize analysis results
        analysis = {
            'total_documents': len(all_documents),
            'news_count': len(custom_data['processed_news']),
            'market_count': len(custom_data['processed_market']),
            'metadata': custom_data['metadata']
        }
        
        # Generate sentiment analysis
        if custom_data['news_data']:
            with st.spinner("🧠 Analyzing news sentiment..."):
                analysis['sentiment_analysis'] = self.advisor.analyze_custom_topic_sentiment(
                    custom_data['news_data'], 
                    custom_data['metadata']['topic'],
                    custom_data['metadata']['country']
                )
        
        # Generate market analysis
        if custom_data['market_data']:
            with st.spinner("📊 Analyzing market trends..."):
                analysis['market_analysis'] = self.advisor.generate_custom_market_analysis(
                    custom_data['market_data'],
                    custom_data['metadata']
                )
        
        return analysis

# =============================================================================
# SAMPLE DATA FOR DEMO MODE
# =============================================================================

def get_sample_data() -> Dict:
    """Sample data for demo mode"""
    return {
        'analysis_results': {
            'total_documents': 45,
            'news_count': 30,
            'market_count': 15,
            'metadata': {
                'topic': 'artificial intelligence technology',
                'country': 'United States',
                'date_range': 'Last 7 Days',
                'region': 'US',
                'fetch_timestamp': datetime.now().isoformat()
            },
            'sentiment_analysis': """
## Overall Sentiment: Cautiously Bullish 📈

Listen up, rookie. The AI sector is showing **cautiously bullish sentiment** right now, which means the market isn't completely losing its mind over shiny new tech toys, but it's not running away screaming either.

### Key Themes (Pay Attention):
1. **Regulatory Scrutiny**: Politicians finally figured out AI exists and are making noise about regulations
2. **Enterprise Adoption**: Companies are actually finding real uses beyond generating cat pictures
3. **Competition Heating Up**: Tech giants are throwing money at AI like it's a Vegas casino

### Market Implications:
- **Short-term**: Volatility as regulations shake out weak players
- **Long-term**: Solid growth for companies with actual AI revenue, not just AI marketing

### Risks You'd Probably Miss:
- Regulatory crackdowns could crush overvalued AI stocks
- AI bubble could pop if reality doesn't match hype
- Competition could eliminate profit margins

**Disclaimer**: This is educational analysis, not personalized investment advice. Don't blame me when you lose money.
            """,
            'market_analysis': """
## Market Performance: Mixed Signals 📊

Alright genius, here's what the numbers are telling us about AI stocks this week:

### Performance Overview:
- **NVDA**: $485.20 (+2.3% this week) - Still the AI darling
- **GOOGL**: $142.15 (-0.8% this week) - Search giant feeling pressure  
- **MSFT**: $378.90 (+1.2% this week) - Azure AI services driving growth
- **AAPL**: $189.25 (-1.1% this week) - Playing catch-up in AI race

### Technical Analysis:
The AI sector is consolidating after the massive run-up. Support levels are holding, but we're seeing profit-taking at resistance levels.

### Fundamental Insights:
- Revenue growth is strong for companies with actual AI products
- Valuations are stretched but not completely insane yet
- Cash flow generation separating winners from pretenders

**Bottom Line**: Stop chasing every AI stock and focus on companies with real revenue and reasonable valuations.

**Disclaimer**: This is educational analysis, not personalized investment advice.
            """
        },
        'processed_data': {
            'news_data': [
                {
                    'title': 'AI Companies Report Strong Q4 Earnings Despite Market Concerns',
                    'description': 'Major artificial intelligence companies exceeded expectations in their latest earnings reports, showing resilience in a challenging market environment.',
                    'source': {'name': 'TechCrunch'},
                    'publishedAt': '2024-01-15T10:30:00Z',
                    'url': 'https://techcrunch.com/ai-earnings'
                },
                {
                    'title': 'Regulators Propose New Framework for AI Governance',
                    'description': 'Government officials unveiled comprehensive guidelines for artificial intelligence oversight, focusing on consumer protection and market stability.',
                    'source': {'name': 'Reuters'},
                    'publishedAt': '2024-01-14T14:22:00Z',
                    'url': 'https://reuters.com/ai-regulation'
                }
            ],
            'market_data': {
                'NVDA': {'close': 485.20, 'change': 2.3},
                'GOOGL': {'close': 142.15, 'change': -0.8},
                'MSFT': {'close': 378.90, 'change': 1.2},
                'AAPL': {'close': 189.25, 'change': -1.1}
            },
            'processed_news': [],
            'processed_market': []
        }
    }

# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

def generate_enhanced_pdf_report(analysis_results: Dict, processed_data: Dict) -> BytesIO:
    """Generate enhanced PDF report with better formatting"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    
    # Title and metadata
    story.append(Paragraph("FinWise AI - Financial Analysis Report", styles['Title']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    metadata = analysis_results['metadata']
    story.append(Paragraph("Executive Summary", styles['Heading1']))
    story.append(Paragraph(f"<b>Analysis Topic:</b> {metadata['topic']}", styles['Normal']))
    story.append(Paragraph(f"<b>Geographic Focus:</b> {metadata.get('country', 'Global')}", styles['Normal']))
    story.append(Paragraph(f"<b>Time Period:</b> {metadata['date_range']}", styles['Normal']))
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.fromisoformat(metadata['fetch_timestamp']).strftime('%B %d, %Y at %H:%M UTC')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Key Metrics
    story.append(Paragraph("Key Metrics", styles['Heading1']))
    metrics_data = [
        ["Metric", "Value", "Description"],
        ["News Articles Analyzed", str(analysis_results['news_count']), "Recent news articles processed"],
        ["Market Data Points", str(analysis_results['market_count']), "Stock/market data points analyzed"],
        ["Total Documents Indexed", str(analysis_results['total_documents']), "Total documents in analysis"],
        ["Unique News Sources", str(len(set([article.get('source', {}).get('name', 'Unknown') 
                                          for article in processed_data['news_data']]))), "Different news outlets covered"],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*72, 1*72, 2.5*72])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Sentiment Analysis
    if 'sentiment_analysis' in analysis_results:
        story.append(Paragraph("News Sentiment Analysis", styles['Heading1']))
        # Clean up the analysis text for PDF
        sentiment_text = analysis_results['sentiment_analysis'].replace('**', '').replace('##', '').replace('### ', '')
        # Split into paragraphs
        for paragraph in sentiment_text.split('\n\n'):
            if paragraph.strip():
                story.append(Paragraph(paragraph.strip(), styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))
    
    # Market Analysis
    if 'market_analysis' in analysis_results:
        story.append(Paragraph("Market Analysis", styles['Heading1']))
        # Clean up the analysis text for PDF
        market_text = analysis_results['market_analysis'].replace('**', '').replace('##', '').replace('### ', '')
        # Split into paragraphs
        for paragraph in market_text.split('\n\n'):
            if paragraph.strip():
                story.append(Paragraph(paragraph.strip(), styles['Normal']))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))
    
    # Top Headlines
    if processed_data['news_data']:
        story.append(Paragraph("Key Headlines Analyzed", styles['Heading1']))
        for i, article in enumerate(processed_data['news_data'][:8], 1):
            headline = f"{i}. {article['title']}"
            if len(headline) > 120:
                headline = headline[:117] + "..."
            
            story.append(Paragraph(f"<b>{headline}</b>", styles['Normal']))
            
            source_info = f"<i>Source: {article.get('source', {}).get('name', 'Unknown')}"
            if article.get('publishedAt'):
                try:
                    pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    source_info += f" | Published: {pub_date.strftime('%B %d, %Y')}"
                except:
                    pass
            source_info += "</i>"
            
            story.append(Paragraph(source_info, styles['Italic']))
            
            if article.get('description'):
                desc = article['description'][:200] + "..." if len(article['description']) > 200 else article['description']
                story.append(Paragraph(desc, styles['Normal']))
            
            story.append(Spacer(1, 8))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph("Report Disclaimer", styles['Heading2']))
    disclaimer_text = """
    This report is generated by FinWise AI for informational and educational purposes only. 
    The analysis is based on publicly available news articles and market data at the time of generation. 
    This report does not constitute investment advice, financial advice, or recommendations to buy or sell any securities. 
    Past performance is not indicative of future results. Always consult with qualified financial professionals 
    before making investment decisions. FinWise AI and its creators are not responsible for any financial 
    losses that may result from decisions based on this report.
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    story.append(Spacer(1, 12))
    story.append(Paragraph("Generated by FinWise AI - Enhanced Financial Intelligence Platform", styles['Italic']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# =============================================================================
# ENHANCED STREAMLIT APPLICATION
# =============================================================================

def create_enhanced_streamlit_app():
    """Create the enhanced Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="FinWise AI Enhanced - Financial Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "FinWise AI Enhanced - Your AI-Powered Financial Intelligence Platform"
        }
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 FinWise AI Enhanced</h1>
        <h3>AI-Powered Financial Intelligence Platform</h3>
        <p>Analyze any topic, anywhere, anytime with enhanced AI insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'api_keys_validated' not in st.session_state:
        st.session_state.api_keys_validated = False
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Demo mode toggle
        demo_mode = st.checkbox(
            "🎮 Demo Mode", 
            value=st.session_state.demo_mode,
            help="Use sample data to explore features without API keys"
        )
        
        if demo_mode != st.session_state.demo_mode:
            st.session_state.demo_mode = demo_mode
            if demo_mode:
                sample_data = get_sample_data()
                st.session_state.update(sample_data)
                st.success("Demo mode activated! Explore with sample data.")
                st.rerun()
        
        st.markdown("---")
        
        # API Help Section
        with st.expander("🔑 How to Get API Keys", expanded=not st.session_state.api_keys_validated):
            st.markdown("""
            ### Required API Keys & Free Tier Limits:
            
            **1. NewsAPI** (Free: 1000 requests/day)
            - Visit: [newsapi.org](https://newsapi.org)
            - Sign up for free API key
            - Perfect for testing and analysis
            
            **2. Alpha Vantage** (Free: 5 calls/min, 500/day)
            - Visit: [alphavantage.co](https://www.alphavantage.co)
            - Free tier includes stock data
            - Sufficient for portfolio analysis
            
            **3. Cohere** (Free: 100 calls/month)
            - Visit: [cohere.ai](https://cohere.ai)
            - Required for AI embeddings
            - Free tier perfect for testing
            
            **4. Groq** (Free tier available)
            - Visit: [groq.com](https://groq.com)
            - Fastest AI inference available
            - Great free tier for chat features
            
            💡 **All APIs offer generous free tiers perfect for testing!**
            """)
        
        # API Configuration
        st.subheader("🔐 API Configuration")
        
        if not demo_mode:
            news_api_key = st.text_input(
                "NewsAPI Key", 
                type="password", 
                help="Get from newsapi.org",
                placeholder="Enter your NewsAPI key"
            )
            alpha_vantage_key = st.text_input(
                "Alpha Vantage Key", 
                type="password", 
                help="Get from alphavantage.co",
                placeholder="Enter your Alpha Vantage key"
            )
            cohere_api_key = st.text_input(
                "Cohere API Key", 
                type="password", 
                help="Get from cohere.ai",
                placeholder="Enter your Cohere API key"
            )
            groq_api_key = st.text_input(
                "Groq API Key", 
                type="password", 
                help="Get from groq.com",
                placeholder="Enter your Groq API key"
            )
            
            # API Validation
            if st.button("🔍 Test API Keys", help="Validate your API keys before analysis"):
                if all([news_api_key, alpha_vantage_key, cohere_api_key, groq_api_key]):
                    with st.spinner("Testing API keys..."):
                        validator = APIValidator()
                        
                        # Test each API
                        news_valid, news_msg = validator.validate_news_api(news_api_key)
                        av_valid, av_msg = validator.validate_alpha_vantage(alpha_vantage_key)
                        cohere_valid, cohere_msg = validator.validate_cohere(cohere_api_key)
                        groq_valid, groq_msg = validator.validate_groq(groq_api_key)
                        
                        # Display results
                        st.write("**Validation Results:**")
                        st.write(news_msg)
                        st.write(av_msg)
                        st.write(cohere_msg)
                        st.write(groq_msg)
                        
                        if all([news_valid, av_valid, cohere_valid, groq_valid]):
                            st.session_state.api_keys_validated = True
                            st.success("🎉 All API keys are valid! Ready for analysis.")
                        else:
                            st.session_state.api_keys_validated = False
                            st.error("❌ Some API keys failed validation. Please check and try again.")
                else:
                    st.warning("Please enter all API keys before testing.")
        else:
            st.info("🎮 Demo mode active - No API keys required!")
            st.session_state.api_keys_validated = True
            news_api_key = alpha_vantage_key = cohere_api_key = groq_api_key = "demo"
        
        st.markdown("---")
        
        # Quick Analysis Presets
        st.subheader("⚡ Quick Analysis Presets")
        preset_options = {
            "Custom Analysis": {"topic": "", "country": "Global", "symbols": []},
            "AI & Technology - US": {
                "topic": "artificial intelligence machine learning technology", 
                "country": "United States", 
                "symbols": ["NVDA", "GOOGL", "MSFT", "AAPL", "TSLA"]
            },
            "Renewable Energy - Global": {
                "topic": "renewable energy solar wind climate", 
                "country": "Global", 
                "symbols": ["ICLN", "PBW", "QCLN", "ENPH", "SEDG"]
            },
            "Healthcare & Biotech": {
                "topic": "healthcare biotechnology pharmaceuticals", 
                "country": "United States", 
                "symbols": ["JNJ", "PFE", "MRNA", "GILD", "AMGN"]
            },
            "Cryptocurrency Markets": {
                "topic": "cryptocurrency bitcoin ethereum blockchain", 
                "country": "Global", 
                "symbols": ["GBTC", "ETHE", "COIN", "MSTR"]
            },
            "ESG & Sustainability": {
                "topic": "ESG sustainability climate change environment", 
                "country": "Global", 
                "symbols": ["ESG", "ICLN", "ESGD", "SUSL"]
            }
        }
        
        selected_preset = st.selectbox(
            "Choose a preset or create custom analysis:",
            list(preset_options.keys()),
            help="Select a pre-configured analysis or choose 'Custom Analysis'"
        )
        
        if selected_preset != "Custom Analysis":
            preset = preset_options[selected_preset]
            # Store preset in session state for main form
            for key, value in preset.items():
                st.session_state[f"preset_{key}"] = value
        
        st.markdown("---")
        
        # Export Options
        st.subheader("📊 Export Options")
        if 'analysis_results' in st.session_state:
            export_format = st.selectbox("Export Format:", ["PDF Report", "JSON Data", "CSV News Data"])
            
            if st.button("📥 Export Analysis", use_container_width=True):
                if export_format == "JSON Data":
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(st.session_state.analysis_results, indent=2, default=str),
                        file_name=f"finwise_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                elif export_format == "CSV News Data" and 'processed_data' in st.session_state:
                    if st.session_state.processed_data.get('news_data'):
                        df = pd.DataFrame([{
                            'Title': article['title'],
                            'Description': article['description'],
                            'Source': article.get('source', {}).get('name', 'Unknown'),
                            'Published': article.get('publishedAt', 'Unknown'),
                            'URL': article.get('url', '')
                        } for article in st.session_state.processed_data['news_data']])
                        
                        st.download_button(
                            "Download CSV",
                            data=df.to_csv(index=False),
                            file_name=f"finwise_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("No news data available for CSV export.")
                elif export_format == "PDF Report" and 'processed_data' in st.session_state:
                    try:
                        with st.spinner("Generating PDF report..."):
                            pdf_buffer = generate_enhanced_pdf_report(
                                st.session_state.analysis_results,
                                st.session_state.processed_data
                            )
                        st.download_button(
                            "Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"finwise_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Failed to generate PDF: {str(e)}")
        else:
            st.info("Run an analysis first to enable export options.")
        
        st.markdown("---")
        
        # Utility buttons
        if st.button("🗑️ Clear Analysis Data", help="Clear current analysis and start fresh"):
            keys_to_clear = ['analysis_results', 'processed_data', 'pipeline', 'chat_history']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Analysis data cleared!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Custom Analysis Configuration")
        
        # Topic and Country
        topic_col, country_col = st.columns(2)
        with topic_col:
            default_topic = st.session_state.get('preset_topic', '')
            analysis_topic = st.text_input(
                "Analysis Topic/Theme",
                value=default_topic,
                placeholder="e.g., artificial intelligence, renewable energy, healthcare innovation",
                help="Enter any topic, sector, or theme you want to analyze"
            )
        
        with country_col:
            countries = ['Global'] + [country for country in EnhancedNewsAPIClient('').country_codes.keys() if country != 'Global']
            default_country = st.session_state.get('preset_country', 'Global')
            selected_country = st.selectbox(
                "Geographic Focus",
                countries,
                index=countries.index(default_country) if default_country in countries else 0,
                help="Choose geographic region for news analysis"
            )
        
        # Time Period Configuration
        st.subheader("⏰ Time Period")
        date_col1, date_col2, date_col3 = st.columns(3)
        
        with date_col1:
            date_preset = st.selectbox(
                "Quick Date Range",
                ["Last 7 Days", "Last 30 Days", "Last 3 Months", "Custom"]
            )
        
        if date_preset != "Custom":
            if date_preset == "Last 7 Days":
                from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
            elif date_preset == "Last 30 Days":
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
            elif date_preset == "Last 3 Months":
                from_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                to_date = datetime.now().strftime('%Y-%m-%d')
        else:
            with date_col2:
                from_date = st.date_input(
                    "From Date", 
                    value=datetime.now() - timedelta(days=30)
                ).strftime('%Y-%m-%d')
            with date_col3:
                to_date = st.date_input(
                    "To Date", 
                    value=datetime.now()
                ).strftime('%Y-%m-%d')
        
        # Market Data Configuration
        st.subheader("📈 Market Data Configuration")
        market_col1, market_col2 = st.columns(2)
        
        with market_col1:
            default_symbols = st.session_state.get('preset_symbols', [])
            stock_symbols_input = st.text_input(
                "Stock Symbols (comma-separated)",
                value=', '.join(default_symbols) if default_symbols else '',
                placeholder="e.g., AAPL, GOOGL, TSLA, MSFT",
                help="Enter stock symbols separated by commas (max 10 symbols)"
            )
            stock_symbols = [symbol.strip().upper() for symbol in stock_symbols_input.split(',') if symbol.strip()] if stock_symbols_input else []
            
            if len(stock_symbols) > 10:
                st.warning("⚠️ Maximum 10 stock symbols allowed to respect API rate limits.")
                stock_symbols = stock_symbols[:10]
        
        with market_col2:
            market_region = st.selectbox(
                "Market Region",
                ["US", "Europe", "Asia", "Emerging", "Global"],
                help="Choose market region for analysis context"
            )
        
        # Advanced Options
        with st.expander("⚙️ Advanced Options"):
            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                news_language = st.selectbox("News Language", ["en", "es", "fr", "de", "it", "pt"])
                sort_by = st.selectbox("Sort News By", ["relevancy", "popularity", "publishedAt"])
            with adv_col2:
                max_articles = st.slider("Max Articles", 20, 100, 50, help="Maximum number of news articles to fetch")
                analysis_depth = st.selectbox("Analysis Depth", ["Standard", "Deep", "Quick"])
        
        # Usage Estimation
        if analysis_topic and stock_symbols:
            st.info(f"""
            📊 **Estimated API Usage for this analysis:**
            - NewsAPI: ~{min(max_articles, 100)} calls
            - Alpha Vantage: ~{len(stock_symbols)} calls  
            - Cohere: ~20-30 calls (embeddings)
            - Groq: ~5-10 calls (AI analysis)
            
            💰 **Cost**: All within free tier limits!
            """)
        
        st.markdown("---")
        
        # Run Analysis Button
        analysis_button_disabled = False
        button_help = "Run comprehensive financial analysis"
        
        if not demo_mode and not st.session_state.api_keys_validated:
            analysis_button_disabled = True
            button_help = "Please validate your API keys first"
        elif not analysis_topic:
            analysis_button_disabled = True
            button_help = "Please enter an analysis topic"
        
        if st.button(
            "🚀 Run Enhanced Analysis", 
            type="primary", 
            use_container_width=True,
            disabled=analysis_button_disabled,
            help=button_help
        ):
            if demo_mode:
                # Use sample data
                sample_data = get_sample_data()
                st.session_state.update(sample_data)
                st.success("""
                🎮 **Demo Analysis Complete!**
                - Using sample data for artificial intelligence topic
                - 30 news articles analyzed
                - 4 stock symbols included
                - Full AI analysis generated
                
                Explore the results in the tabs below!
                """)
            else:
                try:
                    # Create configuration
                    config = APIConfig(
                        news_api_key=news_api_key,
                        alpha_vantage_key=alpha_vantage_key,
                        cohere_api_key=cohere_api_key,
                        groq_api_key=groq_api_key
                    )
                    
                    # Create pipeline
                    pipeline = EnhancedFinWiseDataPipeline(config)
                    
                    # Show progress
                    progress_container = st.container()
                    
                    with progress_container:
                        # Fetch data
                        custom_data = pipeline.fetch_custom_data(
                            topic=analysis_topic,
                            country=selected_country if selected_country != 'Global' else None,
                            from_date=from_date,
                            to_date=to_date,
                            stock_symbols=stock_symbols,
                            region=market_region
                        )
                        
                        # Build analysis
                        st.info("🧠 Building comprehensive analysis...")
                        analysis_results = pipeline.build_custom_analysis(custom_data)
                        
                        # Store results
                        st.session_state.analysis_results = analysis_results
                        st.session_state.processed_data = custom_data
                        st.session_state.pipeline = pipeline
                    
                    # Success message
                    st.success(f"""
                    🎉 **Analysis Complete!**
                    - Processed **{analysis_results['news_count']}** news articles
                    - Analyzed **{analysis_results['market_count']}** market data points
                    - Topic: **{analysis_topic}**
                    - Region: **{selected_country}**
                    - Time Period: **{date_preset}**
                    
                    Explore your results in the tabs below!
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.error("Please check your API keys and internet connection, then try again.")
    
    with col2:
        st.header("📊 Quick Stats")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            # Metrics display
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "📰 News Articles", 
                    results['news_count'],
                    help="Number of news articles analyzed"
                )
                st.metric(
                    "📈 Market Data Points", 
                    results['market_count'],
                    help="Number of market data points processed"
                )
            with col_b:
                st.metric(
                    "📚 Total Documents", 
                    results['total_documents'],
                    help="Total documents indexed for AI search"
                )
                st.metric(
                    "⏰ Last Updated", 
                    datetime.fromisoformat(results['metadata']['fetch_timestamp']).strftime('%H:%M'),
                    help="Time when analysis was completed"
                )
            
            # Current analysis info
            st.markdown(f"""
            <div class="metric-card">
            <h4>📋 Current Analysis</h4>
            <p><strong>Topic:</strong> {results['metadata']['topic']}</p>
            <p><strong>Region:</strong> {results['metadata'].get('country', 'Global')}</p>
            <p><strong>Period:</strong> {results['metadata']['date_range']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.info("🚀 Configure and run your first analysis to see results here!")
            st.markdown("""
            **Getting Started:**
            1. 🔑 Add your API keys (or try Demo Mode)
            2. 📊 Choose a topic to analyze
            3. 🤖 Get AI-powered insights
            
            **Perfect for:**
            - 🎓 Students & researchers
            - 💼 Finance professionals  
            - 📈 Individual investors
            - 🏢 Market analysts
            """)
        
        # Market status
        st.subheader("🏪 Market Status")
        current_time = datetime.now()
        
        # Simple market hours check (US Eastern Time approximation)
        is_weekday = current_time.weekday() < 5
        current_hour = current_time.hour
        
        # Rough approximation of market hours (9:30 AM - 4:00 PM ET)
        market_likely_open = is_weekday and 9 <= current_hour <= 16
        
        if market_likely_open:
            st.success("🟢 US Markets Likely OPEN")
        else:
            st.warning("🔴 US Markets Likely CLOSED")
        
        st.write(f"Current time: {current_time.strftime('%H:%M UTC')}")
        st.caption("Market status is approximate. Check official sources for exact trading hours.")
    
    # Results Display Section
    if 'analysis_results' in st.session_state and 'processed_data' in st.session_state:
        st.markdown("---")
        st.header("📈 Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Overview", 
            "📰 News Analysis", 
            "📊 Market Analysis", 
            "🤖 AI Chat", 
            "📉 Visualizations"
        ])
        
        with tab1:
            st.subheader("Analysis Overview")
            results = st.session_state.analysis_results
            data = st.session_state.processed_data
            
            # Key metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                unique_sources = len(set([
                    article.get('source', {}).get('name', 'Unknown') 
                    for article in data['news_data'] 
                    if article.get('source', {}).get('name')
                ]))
                st.metric("📺 News Sources", unique_sources)
            
            with metrics_col2:
                st.metric("🏢 Market Symbols", len(data['market_data']))
            
            with metrics_col3:
                st.metric("🌍 Geographic Scope", results['metadata'].get('country', 'Global'))
            
            with metrics_col4:
                st.metric("📅 Analysis Timeframe", results['metadata']['date_range'])
            
            # Recent headlines
            if data['news_data']:
                st.subheader("🗞️ Latest Headlines")
                for i, article in enumerate(data['news_data'][:6], 1):
                    with st.expander(f"{i}. {article['title'][:100]}{'...' if len(article['title']) > 100 else ''}"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Description:** {article.get('description', 'No description available')}")
                            if article.get('url'):
                                st.markdown(f"🔗 [Read Full Article]({article['url']})")
                        with col2:
                            st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                            if article.get('publishedAt'):
                                try:
                                    pub_time = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                                    st.write(f"**Published:** {pub_time.strftime('%m/%d %H:%M')}")
                                except:
                                    st.write(f"**Published:** {article.get('publishedAt', 'Unknown')}")
        
        with tab2:
            st.subheader("📰 News Sentiment Analysis")
            
            if 'sentiment_analysis' in st.session_state.analysis_results:
                # Display sentiment analysis with better formatting
                sentiment_text = st.session_state.analysis_results['sentiment_analysis']
                st.markdown(sentiment_text)
            else:
                st.info("No sentiment analysis available. Make sure news data was fetched successfully.")
            
            # News data table
            if st.session_state.processed_data['news_data']:
                st.subheader("📊 News Articles Data")
                
                # Create a more detailed DataFrame
                news_df = pd.DataFrame([{
                    'Title': article['title'][:80] + '...' if len(article['title']) > 80 else article['title'],
                    'Source': article.get('source', {}).get('name', 'Unknown'),
                    'Published': article.get('publishedAt', 'Unknown'),
                    'URL': f"[Link]({article.get('url', '')})" if article.get('url') else 'N/A'
                } for article in st.session_state.processed_data['news_data']])
                
                # Display with filtering options
                sources = ['All'] + sorted(news_df['Source'].unique().tolist())
                selected_source = st.selectbox("Filter by Source:", sources)
                
                if selected_source != 'All':
                    filtered_df = news_df[news_df['Source'] == selected_source]
                else:
                    filtered_df = news_df
                
                st.dataframe(
                    filtered_df, 
                    use_container_width=True,
                    column_config={
                        "URL": st.column_config.LinkColumn("Article Link")
                    }
                )
        
        with tab3:
            st.subheader("📊 Market Data Analysis")
            
            if 'market_analysis' in st.session_state.analysis_results:
                # Display market analysis with better formatting
                market_text = st.session_state.analysis_results['market_analysis']
                st.markdown(market_text)
            else:
                st.info("No market analysis available. Add stock symbols to your analysis.")
            
            # Market data overview
            if st.session_state.processed_data['market_data']:
                st.subheader("📈 Market Data Overview")
                
                market_summary = []
                for symbol, data in st.session_state.processed_data['market_data'].items():
                    if data and isinstance(data, dict):
                        # Find time series key
                        time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
                        if time_series_key and time_series_key in data:
                            time_series = data[time_series_key]
                            recent_dates = list(time_series.keys())[:5]
                            
                            if recent_dates:
                                latest_date = recent_dates[0]
                                latest_data = time_series[latest_date]
                                
                                try:
                                    close_price = float(latest_data.get('4. close', latest_data.get('close', 0)))
                                    volume = latest_data.get('5. volume', 'N/A')
                                    
                                    # Calculate 5-day change if possible
                                    change_pct = 0
                                    if len(recent_dates) >= 2:
                                        old_data = time_series[recent_dates[-1]]
                                        old_close = float(old_data.get('4. close', old_data.get('close', 0)))
                                        if old_close > 0:
                                            change_pct = ((close_price - old_close) / old_close) * 100
                                    
                                    market_summary.append({
                                        'Symbol': symbol,
                                        'Latest Close': f"${close_price:.2f}",
                                        '5-Day Change %': f"{change_pct:+.2f}%",
                                        'Date': latest_date,
                                        'Volume': f"{int(volume):,}" if str(volume).isdigit() else volume
                                    })
                                except (ValueError, TypeError):
                                    market_summary.append({
                                        'Symbol': symbol,
                                        'Latest Close': 'Data Error',
                                        '5-Day Change %': 'N/A',
                                        'Date': latest_date,
                                        'Volume': 'N/A'
                                    })
                
                if market_summary:
                    market_df = pd.DataFrame(market_summary)
                    st.dataframe(market_df, use_container_width=True)
                else:
                    st.warning("No market data could be processed.")
        
        with tab4:
            st.subheader("🤖 AI Financial Assistant")
            
            # Warning message
            st.warning("⚠️ **Disclaimer:** I'm a snarky financial teacher who uses the latest analysis report to answer questions. Make sure your analysis is relevant to get useful responses!")
            
            st.markdown("Ask me anything about your analysis or financial markets in general!")
            
            # Chat history display
            if st.session_state.chat_history:
                st.subheader("💬 Recent Conversations")
                
                # Show recent chats in reverse order (newest first)
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                    with st.container():
                        st.markdown(f"**🙋 You ({chat['timestamp']}):**")
                        st.markdown(f"> {chat['question']}")
                        st.markdown(f"**🤖 FinWise AI:**")
                        st.markdown(chat['answer'])
                        st.markdown("---")
                        
                        if i >= 3:  # Show only last 3 conversations here
                            break
                
                if len(st.session_state.chat_history) > 3:
                    st.info(f"💬 {len(st.session_state.chat_history)} total conversations in this session.")
            else:
                st.info("💭 Start a conversation by asking a question below! I don't hold back, so be ready for some tough love.")
            
            # Sample questions
            with st.expander("💡 Sample Questions to Get Started"):
                st.markdown("""
                **Analysis-Specific Questions:**
                - "What's the sentiment around [your topic] right now?"
                - "Should I invest in [stock from your analysis]?"
                - "What are the key risks I should know about?"
                
                **General Financial Questions:**
                - "Explain P/E ratios like I'm 5 years old"
                - "How do I diversify my portfolio?"
                - "What's the difference between stocks and bonds?"
                
                **Market Questions:**
                - "Why did [stock] drop today?"
                - "Is this a good time to buy tech stocks?"
                - "What should I watch for next week?"
                """)
        
        with tab5:
            st.subheader("📈 Data Visualizations")
            
            # Market performance charts
            if st.session_state.processed_data['market_data']:
                st.subheader("📊 Market Performance Charts")
                
                for symbol, data in st.session_state.processed_data['market_data'].items():
                    if data and isinstance(data, dict):
                        time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
                        if time_series_key and time_series_key in data:
                            time_series = data[time_series_key]
                            
                            # Prepare data for plotting
                            dates = []
                            closes = []
                            volumes = []
                            
                            for date, values in list(time_series.items())[:30]:
                                try:
                                    close_price = float(values.get('4. close', values.get('close', 0)))
                                    volume = int(values.get('5. volume', 0))
                                    
                                    dates.append(pd.to_datetime(date))
                                    closes.append(close_price)
                                    volumes.append(volume)
                                except (ValueError, TypeError):
                                    continue
                            
                            if dates and closes:
                                # Price chart
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=dates[::-1],  # Reverse to show chronologically
                                    y=closes[::-1],
                                    mode='lines+markers',
                                    name=f'{symbol} Price',
                                    line=dict(color='#1f77b4', width=2),
                                    hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
                                ))
                                
                                fig.update_layout(
                                    title=f'{symbol} - Price Movement (Last 30 Days)',
                                    xaxis_title='Date',
                                    yaxis_title='Price ($)',
                                    hovermode='x unified',
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{symbol}")
            
            # News sources distribution
            if st.session_state.processed_data['news_data']:
                st.subheader("📺 News Sources Distribution")
                
                sources = [article.get('source', {}).get('name', 'Unknown')
                          for article in st.session_state.processed_data['news_data']]
                source_counts = pd.Series(sources).value_counts().head(10)
                
                fig = px.bar(
                    x=source_counts.values,
                    y=source_counts.index,
                    orientation='h',
                    title="Top News Sources by Article Count",
                    labels={'x': 'Number of Articles', 'y': 'Source'},
                    color=source_counts.values,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                # News timeline
                st.subheader("📅 News Timeline")
                news_dates = []
                for article in st.session_state.processed_data['news_data']:
                    if article.get('publishedAt'):
                        try:
                            pub_date = pd.to_datetime(article['publishedAt']).date()
                            news_dates.append(pub_date)
                        except:
                            continue
                
                if news_dates:
                    date_counts = pd.Series(news_dates).value_counts().sort_index()
                    
                    fig = px.line(
                        x=date_counts.index,
                        y=date_counts.values,
                        title="News Articles Published Over Time",
                        labels={'x': 'Date', 'y': 'Number of Articles'},
                        markers=True
                    )
                    fig.update_layout(template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Chat Input Section
    if 'analysis_results' in st.session_state and 'processed_data' in st.session_state:
        st.markdown("---")
        st.subheader("💬 Ask FinWise AI")
        
        # Initialize processing state
        if 'processing_query' not in st.session_state:
            st.session_state.processing_query = False
        
        # Chat input
        user_question = st.chat_input(
            "Ask a financial question about your analysis... (Warning: I don't hold back!)",
            disabled=st.session_state.processing_query
        )
        
        if (user_question and 'pipeline' in st.session_state and 
            not st.session_state.processing_query and
            (not st.session_state.chat_history or 
             st.session_state.chat_history[-1]['question'] != user_question)):
            
            st.session_state.processing_query = True
            
            with st.spinner("🤔 Analyzing your question... Don't expect me to go easy on you!"):
                try:
                    # Prepare filters for search
                    filters = {
                        'topic': st.session_state.analysis_results['metadata']['topic'],
                        'country': st.session_state.analysis_results['metadata'].get('country')
                    }
                    filters = {k: v for k, v in filters.items() if v is not None}
                    
                    # Get AI response
                    answer = st.session_state.pipeline.advisor.answer_custom_financial_query(
                        user_question, filters
                    )
                    
                    # Store conversation
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
                    
                    st.success("✅ Response generated! Check the AI Chat tab or scroll up to see the conversation.")
                    
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
                finally:
                    st.session_state.processing_query = False
                    st.rerun()
        
        elif user_question and not st.session_state.get('pipeline'):
            st.warning("⚠️ Run an analysis first to enable the AI assistant!")
        
        # Show latest conversation if available
        if st.session_state.chat_history:
            with st.expander("💬 Latest Conversation", expanded=True):
                latest_chat = st.session_state.chat_history[-1]
                st.markdown(f"**🙋 You ({latest_chat['timestamp']}):**")
                st.markdown(f"> {latest_chat['question']}")
                st.markdown(f"**🤖 FinWise AI:**")
                st.markdown(latest_chat['answer'])
                
                if len(st.session_state.chat_history) > 1:
                    st.info(f"💬 View all {len(st.session_state.chat_history)} conversations in the AI Chat tab above.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🚀 FinWise AI Enhanced</h3>
        <p><strong>AI-Powered Global Financial Intelligence Platform</strong></p>
        <p>Analyze any topic, anywhere, anytime with enhanced AI insights</p>
        <p style='margin-top: 1rem;'><small>⚠️ <strong>Important Disclaimer:</strong> This tool provides informational analysis only and is not financial advice. 
        All analysis is based on publicly available data and AI interpretation. Always consult with qualified 
        financial professionals before making investment decisions. Past performance does not guarantee future results.</small></p>
        <p style='margin-top: 1rem;'><strong>Created with ❤️ for the financial community</strong></p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    create_enhanced_streamlit_app()
