import os
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
import json
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

@dataclass
class APIConfig:
    news_api_key: str
    alpha_vantage_key: str
    cohere_api_key: str
    groq_api_key: Optional[str] = None
    base_news_url: str = "https://newsapi.org/v2"
    base_av_url: str = "https://www.alphavantage.co/query"

class EnhancedNewsAPIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
        self.country_codes = {
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
            'general', 'business', 'entertainment', 'health', 'science', 
            'sports', 'technology'
        ]
    
    def get_custom_news(self, 
                       topic: str, 
                       country: str = None, 
                       category: str = None,
                       from_date: str = None, 
                       to_date: str = None,
                       language: str = 'en',
                       sort_by: str = 'relevancy', 
                       page_size: int = 50) -> List[Dict]:
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
            if country and country in self.country_codes:
                params['country'] = self.country_codes[country]
            if category and category in self.categories:
                params['category'] = category
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                return [article for article in articles if article.get('title') and article.get('description')]
            else:
                st.error(f"News API Error: {data.get('message', 'Unknown error')}")
                return []
        except requests.exceptions.RequestException as e:
            st.error(f"Request Error: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return []
    
    def get_trending_topics(self, country: str = 'us') -> List[Dict]:
        country_code = self.country_codes.get(country, 'us')
        params = {
            'country': country_code,
            'apiKey': self.api_key,
            'pageSize': 20
        }
        try:
            response = requests.get(f"{self.base_url}/top-headlines", params=params)
            if response.status_code == 200:
                return response.json().get('articles', [])
        except Exception as e:
            st.error(f"Error fetching trending topics: {str(e)}")
        return []
    
    def search_sources(self, country: str = None, category: str = None) -> List[Dict]:
        params = {'apiKey': self.api_key}
        if country and country in self.country_codes:
            params['country'] = self.country_codes[country]
        if category and category in self.categories:
            params['category'] = category
        try:
            response = requests.get(f"{self.base_url}/sources", params=params)
            if response.status_code == 200:
                return response.json().get('sources', [])
        except Exception as e:
            st.error(f"Error fetching sources: {str(e)}")
        return []

class EnhancedAlphaVantageClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
        self.global_indices = {
            'US': ['SPY', 'QQQ', 'DIA', 'IWM'],
            'Europe': ['EWG', 'EWU', 'EWQ', 'EWI'],
            'Asia': ['EWJ', 'EWY', 'INDA', 'EWS'],
            'Emerging': ['EEM', 'VWO', 'EWZ', 'RSX']
        }
        
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrial': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Communication': 'XLC'
        }
    
    def get_custom_market_data(self, 
                              symbols: List[str], 
                              interval: str = "daily",
                              region: str = "US") -> Dict[str, Dict]:
        results = {}
        function_map = {
            "daily": "TIME_SERIES_DAILY",
            "weekly": "TIME_SERIES_WEEKLY",  
            "monthly": "TIME_SERIES_MONTHLY",
            "intraday": "TIME_SERIES_INTRADAY"
        }
        for symbol in symbols:
            try:
                params = {
                    'function': function_map.get(interval, "TIME_SERIES_DAILY"),
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'compact'
                }
                if interval == "intraday":
                    params['interval'] = '60min'
                response = requests.get(self.base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if "Error Message" in data:
                        st.warning(f"Error for {symbol}: {data['Error Message']}")
                        continue
                    elif "Note" in data:
                        st.warning(f"API limit reached for {symbol}: {data['Note']}")
                        continue
                    results[symbol] = data
                else:
                    st.warning(f"Failed to fetch data for {symbol}: HTTP {response.status_code}")
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        return results
    
    def get_regional_indices(self, region: str) -> Dict[str, Dict]:
        if region in self.global_indices:
            symbols = self.global_indices[region]
            return self.get_custom_market_data(symbols)
        return {}
    
    def get_sector_analysis(self, sectors: List[str]) -> Dict[str, Dict]:
        symbols = [self.sector_etfs[sector] for sector in sectors if sector in self.sector_etfs]
        return self.get_custom_market_data(symbols)
    
    def get_forex_data(self, from_currency: str = "USD", to_currency: str = "EUR") -> Dict:
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_currency.upper(),
            'to_symbol': to_currency.upper(),
            'apikey': self.api_key
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching forex data: {str(e)}")
        return {}
    
    def get_economic_indicators(self, indicator: str = "GDP", country: str = "US") -> Dict:
        indicators_map = {
            'GDP': 'REAL_GDP',
            'CPI': 'CPI', 
            'Unemployment': 'UNEMPLOYMENT',
            'Federal Funds Rate': 'FEDERAL_FUNDS_RATE',
            'Treasury Yield': 'TREASURY_YIELD'
        }
        function_name = indicators_map.get(indicator, 'REAL_GDP')
        params = {
            'function': function_name,
            'apikey': self.api_key
        }
        if function_name in ['CPI', 'UNEMPLOYMENT']:
            params['interval'] = 'monthly'
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching economic data: {str(e)}")
        return {}

class EnhancedFinancialDataProcessor:
    def __init__(self, cohere_api_key: str):
        self.cohere_client = cohere.Client(cohere_api_key)
        self.index = None
        self.documents = []
    
    def process_custom_news_data(self, articles: List[Dict], topic: str, country: str = None) -> List[Dict]:
        processed = []
        for article in articles:
            if article.get('title') and article.get('description'):
                content_parts = [
                    f"Topic: {topic}",
                    f"Title: {article['title']}",
                    f"Description: {article['description']}"
                ]
                if country:
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
                except (ValueError, KeyError) as e:
                    continue
        return processed
    
    def build_enhanced_vector_index(self, documents: List[Dict]):
        if not documents:
            st.warning("No documents to index")
            return
        self.documents = documents
        texts = [doc['content'] for doc in documents if doc.get('content')]
        if not texts:
            st.warning("No valid content found in documents")
            return
        try:
            batch_size = 96
            all_embeddings = []
            with st.spinner(f"Generating embeddings for {len(texts)} documents..."):
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    response = self.cohere_client.embed(
                        texts=batch_texts,
                        model="embed-english-v3.0",
                        input_type="search_document"
                    )
                    all_embeddings.extend(response.embeddings)
            embeddings = np.array(all_embeddings)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            st.success(f"Successfully indexed {len(texts)} documents")
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return
    
    def search_enhanced_similar(self, query: str, k: int = 5, filters: Dict = None) -> List[Tuple[Dict, float]]:
        if not self.index or not self.documents:
            return []
        try:
            response = self.cohere_client.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query"
            )
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
            st.error(f"Error during search: {str(e)}")
            return []

class EnhancedFinancialAdvisor:
    def __init__(self, groq_api_key: str, cohere_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.data_processor = EnhancedFinancialDataProcessor(cohere_api_key)
        self.conversation_history = []
    
    def _call_groq_with_context(self, prompt: str, context: Dict, model: str = "llama3-8b-8192") -> str:
        try:
            system_message = f"""You are FinWise AI, a sarcastic and brutally honest financial teacher who doesn’t sugarcoat things. You’re here to educate users like they’re in a tough finance class, throwing in sharp wit and zero patience for dumb questions. Your job is to:
1. Teach financial concepts with clear explanations, analogies, and examples.
2. Answer questions directly, but don’t hold back on pointing out obvious mistakes or gaps in the user’s thinking.
3. Use the latest analysis report to ground your answers when relevant (matches the topic and is fresh within 24 hours). If it’s not, tell the user why their request is pointless.
4. Use recent conversation history to stay on track, but don’t bore them with old stuff.
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

Be rude, but professional. Call out nonsense, use sarcasm, and make it educational. Always end with a disclaimer that this isn’t personalized financial advice."""
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.5,
                max_tokens=1500
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _get_conversation_summary(self, topic: str) -> str:
        """Summarize last 3 relevant conversations for context."""
        if not self.conversation_history:
            return "No prior conversations. You’re starting from scratch, genius."
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
        """Summarize the latest analysis report if relevant."""
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
        if not news_data:
            return "No news data available. Did you even run an analysis, or are you just wasting my time?"
        news_content = "\n".join([
            f"- {article['title']}: {article['description'][:200]}..." 
            for article in news_data[:15]
        ])
        geographic_context = f" in {country}" if country else " globally"
        prompt = f"""
        Analyze the sentiment around "{topic}"{geographic_context} based on this news:
        
        {news_content}
        
        As a snarky financial teacher, do this:
        1. Give the overall sentiment (bullish/bearish/neutral) and explain it like I’m a rookie who barely gets it.
        2. List key themes and developments, with examples to hammer it into my head.
        3. Highlight {country if country else 'global'} factors, because apparently I need to know this.
        4. Spell out market implications for {topic}-related investments, since I can’t figure it out myself.
        5. Point out risks and opportunities, because I’d probably miss them.
        6. Tell me what to watch next, so I don’t screw this up.
        
        Keep it under 600 words. Educate me, don’t coddle me.
        """
        context = {
            'topic': topic,
            'country': country,
            'data_sources': 'Recent News Articles'
        }
        return self._call_groq_with_context(prompt, context)
    
    def generate_custom_market_analysis(self, market_data: Dict, context: Dict) -> str:
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
        
        You’re a financial teacher with zero patience. Do this:
        1. Give a regional performance overview, like I don’t know what a stock chart is.
        2. Explain {topic}-specific trends, with analogies so I get it.
        3. Compare instruments, because I’m too lazy to do it myself.
        4. Highlight {region} market factors, since I apparently need a geography lesson.
        5. Provide technical and fundamental insights, or I’ll miss the obvious.
        6. List risks and opportunities, because I’d probably tank my portfolio.
        7. Suggest next steps for {region} exposure, if I’m not totally hopeless.
        
        Educate me in under 600 words. Don’t waste my time.
        """
        return self._call_groq_with_context(prompt, context)
    
    def answer_custom_financial_query(self, query: str, filters: Dict = None) -> str:
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
        
        You’re a rude financial teacher. Answer like I’m a student who didn’t do the reading:
        1. **Direct Answer**: Answer the question bluntly. Call out any dumb assumptions.
        2. **Financial Lesson**: Explain a related financial concept with an analogy and example.
        3. **Report Insights**: Summarize key findings from the analysis report if relevant.
        4. **Evidence**: Use the report or similar data to back it up. Don’t make stuff up.
        5. **Market Context**: Tie it to broader markets, since I clearly don’t get it.
        6. **Risks**: List what could go wrong, because I’d probably ignore them.
        7. **Advice (if relevant)**: {'Give actionable advice based on the analysis report.' if advice_eligible else f'No advice because: {advice_reason}'}
        
        Data Source: {source_info}
        Keep it under 600 words. Educate me, don’t bore me.
        """
        
        analysis_context = {
            'topic': filters.get('topic', 'Custom Query') if filters else 'Custom Query',
            'country': filters.get('country', 'Global') if filters else 'Global',
            'data_sources': source_info
        }
        
        response = self._call_groq_with_context(prompt, analysis_context)
        
        self.conversation_history.append({
            'query': query,
            'response': response,
            'filters': filters,
            'timestamp': datetime.now().isoformat()
        })
        
        return response

class EnhancedFinWiseDataPipeline:
    def __init__(self, config: APIConfig, groq_api_key: str):
        self.news_client = EnhancedNewsAPIClient(config.news_api_key)
        self.av_client = EnhancedAlphaVantageClient(config.alpha_vantage_key)
        self.data_processor = EnhancedFinancialDataProcessor(config.cohere_api_key)
        self.advisor = EnhancedFinancialAdvisor(groq_api_key, config.cohere_api_key)
    
    def fetch_custom_data(self, 
                         topic: str,
                         country: str = None,
                         from_date: str = None,
                         to_date: str = None,
                         stock_symbols: List[str] = None,
                         region: str = "US") -> Dict:
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
        with st.spinner(f"Fetching news for '{topic}'..."):
            news_data = self.news_client.get_custom_news(
                topic=topic,
                country=country,
                from_date=from_date,
                to_date=to_date
            )
            results['news_data'] = news_data
        if news_data:
            results['processed_news'] = self.data_processor.process_custom_news_data(
                news_data, topic, country
            )
        if stock_symbols:
            with st.spinner(f"Fetching market data for {len(stock_symbols)} symbols..."):
                market_data = self.av_client.get_custom_market_data(stock_symbols, region=region)
                results['market_data'] = market_data
            if market_data:
                market_context = {'region': region, 'topic': topic}
                results['processed_market'] = self.data_processor.process_enhanced_market_data(
                    market_data, market_context
                )
        return results
    
    def build_custom_analysis(self, custom_data: Dict) -> Dict:
        all_documents = []
        if custom_data['processed_news']:
            all_documents.extend(custom_data['processed_news'])
        if custom_data['processed_market']:
            all_documents.extend(custom_data['processed_market'])
        if all_documents:
            self.data_processor.build_enhanced_vector_index(all_documents)
        analysis = {
            'total_documents': len(all_documents),
            'news_count': len(custom_data['processed_news']),
            'market_count': len(custom_data['processed_market']),
            'metadata': custom_data['metadata']
        }
        if custom_data['news_data']:
            analysis['sentiment_analysis'] = self.advisor.analyze_custom_topic_sentiment(
                custom_data['news_data'], 
                custom_data['metadata']['topic'],
                custom_data['metadata']['country']
            )
        if custom_data['market_data']:
            analysis['market_analysis'] = self.advisor.generate_custom_market_analysis(
                custom_data['market_data'],
                custom_data['metadata']
            )
        return analysis

def generate_pdf_report(analysis_results, processed_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("FinWise AI Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    metadata = analysis_results['metadata']
    story.append(Paragraph(f"Topic: {metadata['topic']}", styles['Heading2']))
    story.append(Paragraph(f"Region: {metadata.get('country', 'Global')}", styles['Normal']))
    story.append(Paragraph(f"Time Period: {metadata['date_range']}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.fromisoformat(metadata['fetch_timestamp']).strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Summary Metrics", styles['Heading2']))
    metrics_data = [
        ["Metric", "Value"],
        ["News Articles", str(analysis_results['news_count'])],
        ["Market Data Points", str(analysis_results['market_count'])],
        ["Total Documents", str(analysis_results['total_documents'])],
        ["Sources", str(len(set([article.get('source', {}).get('name', 'Unknown') 
                                for article in processed_data['news_data']])))]
    ]
    metrics_table = Table(metrics_data)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))
    if 'sentiment_analysis' in analysis_results:
        story.append(Paragraph("News Sentiment Analysis", styles['Heading2']))
        story.append(Paragraph(analysis_results['sentiment_analysis'].replace('\n', '<br/>'), 
                             styles['Normal']))
        story.append(Spacer(1, 12))
    if 'market_analysis' in analysis_results:
        story.append(Paragraph("Market Analysis", styles['Heading2']))
        story.append(Paragraph(analysis_results['market_analysis'].replace('\n', '<br/>'), 
                             styles['Normal']))
        story.append(Spacer(1, 12))
    if processed_data['news_data']:
        story.append(Paragraph("Latest Headlines", styles['Heading2']))
        for i, article in enumerate(processed_data['news_data'][:5]):
            headline = f"{i+1}. {article['title'][:100]}{'...' if len(article['title']) > 100 else ''}"
            source = article.get('source', {}).get('name', 'Unknown')
            published = article.get('publishedAt', 'Unknown')
            story.append(Paragraph(headline, styles['Normal']))
            story.append(Paragraph(f"Source: {source} | Published: {published}", 
                                 styles['Italic']))
            story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_enhanced_streamlit_app():
    st.set_page_config(
        page_title="FinWise AI - Financial Intelligence",
        page_icon="📈",
        layout="wide"
    )
    st.title("FinWise AI - Financial Intelligence System")
    st.markdown("*Analyze any topic, anywhere, anytime with AI-powered insights*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Configuration")
        if st.button("Clear Data", help="Clear analysis data only"):
            try:
                analysis_keys = ['analysis_results', 'processed_data', 'pipeline', 'chat_history']
                for key in analysis_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("Analysis data cleared!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        st.subheader("API Configuration")
        news_api_key = st.text_input("News API Key", type="password", help="Get from newsapi.org")
        alpha_vantage_key = st.text_input("Alpha Vantage Key", type="password", help="Get from alphavantage.co")
        cohere_api_key = st.text_input("Cohere API Key", type="password", help="Get from cohere.ai")
        groq_api_key = st.text_input("Groq API Key", type="password", help="Get from groq.com")
        st.markdown("---")
        st.subheader("Quick Analysis Presets")
        preset_options = {
            "Technology Trends - US": {"topic": "artificial intelligence technology trends", "country": "United States", "symbols": ["AAPL", "GOOGL", "MSFT", "NVDA"]},
            "Renewable Energy - Europe": {"topic": "renewable energy solar wind", "country": "Germany", "symbols": ["ICLN", "PBW", "QCLN"]},
            "Healthcare Innovation - Global": {"topic": "healthcare biotechnology innovation", "country": None, "symbols": ["JNJ", "PFE", "MRNA", "GILD"]},
            "Cryptocurrency - Asia": {"topic": "cryptocurrency bitcoin regulation", "country": "Japan", "symbols": ["GBTC", "ETHE"]},
            "Climate Change Impact": {"topic": "climate change environmental policy", "country": "Canada", "symbols": ["ESG", "ICLN"]}
        }
        selected_preset = st.selectbox("Choose Preset:", ["Custom Analysis"] + list(preset_options.keys()))
        if selected_preset != "Custom Analysis":
            preset = preset_options[selected_preset]
            st.session_state.update(preset)
        st.markdown("---")
        st.subheader("Export Options")
        export_format = st.selectbox("Export Format:", ["JSON", "CSV", "PDF Report"])
        if st.button("Export Analysis Data"):
            if 'analysis_results' in st.session_state:
                if export_format == "JSON":
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(st.session_state.analysis_results, indent=2),
                        file_name=f"finwise_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                elif export_format == "CSV" and 'processed_data' in st.session_state:
                    df = pd.DataFrame(st.session_state.processed_data.get('processed_news', []))
                    st.download_button(
                        "Download CSV",
                        data=df.to_csv(index=False),
                        file_name=f"finwise_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                elif export_format == "PDF Report" and 'processed_data' in st.session_state:
                    try:
                        pdf_buffer = generate_pdf_report(
                            st.session_state.analysis_results,
                            st.session_state.processed_data
                        )
                        st.download_button(
                            "Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"finwise_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Failed to generate PDF: {str(e)}")
                else:
                    st.warning("No data available for PDF export. Run an analysis first!")
            else:
                st.warning("No analysis data to export. Run an analysis first!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Custom Analysis Configuration")
        topic_col, country_col = st.columns(2)
        with topic_col:
            analysis_topic = st.text_input(
                "Analysis Topic/Theme",
                value=st.session_state.get('topic', ''),
                placeholder="e.g., artificial intelligence, renewable energy, healthcare innovation",
                help="Enter any topic, sector, or theme you want to analyze"
            )
        with country_col:
            countries = ['Global'] + list(EnhancedNewsAPIClient('').country_codes.keys())
            selected_country = st.selectbox(
                "Geographic Focus",
                countries,
                index=countries.index(st.session_state.get('country', 'Global')) if st.session_state.get('country') in countries else 0
            )
        st.subheader("Time Period")
        date_col1, date_col2, date_col3 = st.columns(3)
        with date_col1:
            date_preset = st.selectbox(
                "Quick Date Range",
                ["Custom", "Last 7 Days", "Last 30 Days", "Last 3 Months"]
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
                from_date = st.date_input("From Date", value=datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            with date_col3:
                to_date = st.date_input("To Date", value=datetime.now()).strftime('%Y-%m-%d')
        st.subheader("Market Data Configuration")
        market_col1, market_col2 = st.columns(2)
        with market_col1:
            stock_symbols_input = st.text_input(
                "Stock Symbols (comma-separated)",
                value=', '.join(st.session_state.get('symbols', [])) if st.session_state.get('symbols') else '',
                placeholder="e.g., AAPL, GOOGL, TSLA, MSFT",
                help="Enter stock symbols separated by commas"
            )
            stock_symbols = [symbol.strip().upper() for symbol in stock_symbols_input.split(',') if symbol.strip()] if stock_symbols_input else []
        with market_col2:
            market_region = st.selectbox(
                "Market Region",
                ["US", "Europe", "Asia", "Emerging", "Global"]
            )
        with st.expander("Advanced Options"):
            adv_col1, adv_col2 = st.columns(2)
            with adv_col1:
                news_language = st.selectbox("News Language", ["en", "es", "fr", "de", "it", "pt"])
                sort_by = st.selectbox("Sort News By", ["relevancy", "popularity", "publishedAt"])
            with adv_col2:
                max_articles = st.slider("Max Articles", 20, 100, 50)
                analysis_depth = st.selectbox("Analysis Depth", ["Standard", "Deep", "Quick"])
        st.markdown("---")
        if st.button("Run Enhanced Analysis", type="primary", use_container_width=True):
            if not all([news_api_key, alpha_vantage_key, cohere_api_key, groq_api_key]):
                st.error("Please provide all required API keys in the sidebar!")
            elif not analysis_topic:
                st.error("Please enter an analysis topic!")
            else:
                config = APIConfig(
                    news_api_key=news_api_key,
                    alpha_vantage_key=alpha_vantage_key,
                    cohere_api_key=cohere_api_key,
                    groq_api_key=groq_api_key
                )
                pipeline = EnhancedFinWiseDataPipeline(config, groq_api_key)
                progress_bar = st.progress(0)
                status_text = st.empty()
                try:
                    status_text.text("Fetching custom data...")
                    progress_bar.progress(20)
                    custom_data = pipeline.fetch_custom_data(
                        topic=analysis_topic,
                        country=selected_country if selected_country != 'Global' else None,
                        from_date=from_date,
                        to_date=to_date,
                        stock_symbols=stock_symbols,
                        region=market_region
                    )
                    progress_bar.progress(50)
                    status_text.text("Building analysis...")
                    analysis_results = pipeline.build_custom_analysis(custom_data)
                    progress_bar.progress(80)
                    status_text.text("Analysis complete!")
                    st.session_state.analysis_results = analysis_results
                    st.session_state.processed_data = custom_data
                    st.session_state.pipeline = pipeline
                    progress_bar.progress(100)
                    status_text.text("Ready to explore results!")
                    st.success(f"""
                    **Analysis Complete!**
                    - Processed {analysis_results['news_count']} news articles
                    - Analyzed {analysis_results['market_count']} market data points
                    - Topic: {analysis_topic}
                    - Region: {selected_country}
                    """)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error("Please check your API keys and try again.")
    
    with col2:
        st.header("Quick Stats")
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("News Articles", results['news_count'])
                st.metric("Market Points", results['market_count'])
            with col_b:
                st.metric("Total Documents", results['total_documents'])
                st.metric("Last Updated", datetime.fromisoformat(results['metadata']['fetch_timestamp']).strftime('%H:%M'))
            st.info(f"""
            **Current Analysis:**
            - **Topic:** {results['metadata']['topic']}
            - **Region:** {results['metadata'].get('country', 'Global')}
            - **Period:** {results['metadata']['date_range']}
            """)
        else:
            st.info("Configure and run your first analysis to see results here!")
        st.subheader("Market Status")
        if st.button("Refresh Market Status"):
            current_time = datetime.now()
            market_hours = 9.5 <= current_time.hour <= 16 and current_time.weekday() < 5
            if market_hours:
                st.success("Markets are OPEN")
            else:
                st.warning("Markets are CLOSED")
            st.write(f"Current time: {current_time.strftime('%H:%M %Z')}")
    
    if 'analysis_results' in st.session_state and 'processed_data' in st.session_state:
        st.markdown("---")
        st.header("Analysis Results")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([" Overview", " News Analysis", " Market Analysis", " AI Chat", " Visualizations"])
        
        with tab1:
            st.subheader("Analysis Overview")
            results = st.session_state.analysis_results
            data = st.session_state.processed_data
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                st.metric("News Sources", len(set([
                    article.get('source', {}).get('name', 'Unknown') 
                    for article in data['news_data'] 
                        if article.get('source', {}).get('name')
                ])))
            with metrics_col2:
                st.metric("Market Symbols", len(data['market_data']))
            with metrics_col3:
                st.metric("Geographic Scope", results['metadata'].get('country', 'Global'))
            with metrics_col4:
                st.metric("Analysis Timeframe", results['metadata']['date_range'])
            if data['news_data']:
                st.subheader(" Latest Headlines")
                for i, article in enumerate(data['news_data'][:5]):
                    with st.expander(f"{article['title'][:80]}..."):
                        st.write(f"**Source:** {article.get('source', {}).get('name', 'Unknown')}")
                        st.write(f"**Published:** {article.get('publishedAt', 'Unknown')}")
                        st.write(f"**Description:** {article.get('description', 'No description available')}")
                        if article.get('url'):
                            st.markdown(f"**[ Read Full Article]({article['url']})**")
        
        with tab2:
            st.subheader(" News Sentiment Analysis")
            if 'sentiment_analysis' in st.session_state.analysis_results:
                st.markdown(st.session_state.analysis_results['sentiment_analysis'])
            else:
                st.info("No sentiment analysis available. Make sure news data was fetched successfully.")
            if st.session_state.processed_data['news_data']:
                st.subheader("News Articles Data")
                news_df = pd.DataFrame([{
                    'Title': article['title'][:100] + '...' if len(article['title']) > 100 else article['title'],
                    'Source': article.get('source', {}).get('name', 'Unknown'),
                    'Published': article.get('publishedAt', 'Unknown'),
                    'URL': article.get('url', '')
                } for article in st.session_state.processed_data['news_data']])
                st.dataframe(news_df, use_container_width=True)
        
        with tab3:
            st.subheader(" Market Data Analysis")
            if 'market_analysis' in st.session_state.analysis_results:
                st.markdown(st.session_state.analysis_results['market_analysis'])
            else:
                st.info("No market analysis available. Add stock symbols to your analysis.")
            if st.session_state.processed_data['market_data']:
                st.subheader(" Market Data Overview")
                market_summary = []
                for symbol, data in st.session_state.processed_data['market_data'].items():
                    if data and isinstance(data, dict):
                        time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
                        if time_series_key and time_series_key in data:
                            time_series = data[time_series_key]
                            recent_date = list(time_series.keys())[0]
                            recent_data = time_series[recent_date]
                            try:
                                close_price = float(recent_data.get('4. close', recent_data.get('close', 0)))
                                volume = recent_data.get('5. volume', 'N/A')
                                market_summary.append({
                                    'Symbol': symbol,
                                    'Latest Close': f"${close_price:.2f}",
                                    'Date': recent_date,
                                    'Volume': volume
                                })
                            except (ValueError, TypeError):
                                continue
                if market_summary:
                    market_df = pd.DataFrame(market_summary)
                    st.dataframe(market_df, use_container_width=True)
        
        with tab4:
            st.subheader(" AI Financial Assistant")
            st.warning("Beware: I'm a snarky financial teacher who doesn’t hold hands. I use the latest analysis report to answer, so make sure it’s relevant!")
            st.markdown("Ask me anything about your analysis or financial markets!")
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if st.session_state.chat_history:
                st.subheader(" Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.container():
                        st.markdown(f"**You ({chat['timestamp']}):** {chat['question']}")
                        st.markdown(f"**FinWise AI:** {chat['answer']}")
                        st.markdown("---")
            else:
                st.info("Start a conversation by asking a question below! Don’t waste my time with nonsense.")
        
        with tab5:
            st.subheader(" Data Visualizations")
            if st.session_state.processed_data['market_data']:
                st.subheader(" Market Performance Charts")
                for symbol, data in st.session_state.processed_data['market_data'].items():
                    if data and isinstance(data, dict):
                        time_series_key = next((k for k in data.keys() if 'Time Series' in k), None)
                        if time_series_key and time_series_key in data:
                            time_series = data[time_series_key]
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
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=dates[::-1],
                                    y=closes[::-1],
                                    mode='lines+markers',
                                    name=f'{symbol} Price',
                                    line=dict(color='blue', width=2)
                                ))
                                fig.update_layout(
                                    title=f'{symbol} - Price Movement',
                                    xaxis_title='Date',
                                    yaxis_title='Price ($)',
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{symbol}")
            if st.session_state.processed_data['news_data']:
                st.subheader("📊 News Sources Distribution")
                sources = [article.get('source', {}).get('name', 'Unknown')
                          for article in st.session_state.processed_data['news_data']]
                source_counts = pd.Series(sources).value_counts().head(10)
                fig = px.bar(
                    x=source_counts.values,
                    y=source_counts.index,
                    orientation='h',
                    title="Top News Sources",
                    labels={'x': 'Number of Articles', 'y': 'Source'}
                )
                st.plotly_chart(fig, use_container_width=True, key="news_sources_chart")
            if st.session_state.processed_data['news_data']:
                st.subheader("News Timeline")
                news_dates = []
                for article in st.session_state.processed_data['news_data']:
                    if article.get('publishedAt'):
                        try:
                            news_dates.append(pd.to_datetime(article['publishedAt']).date())
                        except:
                            continue
                if news_dates:
                    date_counts = pd.Series(news_dates).value_counts().sort_index()
                    fig = px.line(
                        x=date_counts.index,
                        y=date_counts.values,
                        title="News Articles Over Time",
                        labels={'x': 'Date', 'y': 'Number of Articles'}
                    )
                    st.plotly_chart(fig, use_container_width=True, key="news_timeline_chart")
    
    if 'analysis_results' in st.session_state and 'processed_data' in st.session_state:
        st.markdown("---")
        st.subheader("Ask FinWise AI")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'processing_query' not in st.session_state:
            st.session_state.processing_query = False
        user_question = st.chat_input("Ask a financial question about your analysis... (I’m not here to coddle you!)")
        if (user_question and 'pipeline' in st.session_state and 
            not st.session_state.processing_query and
            (not st.session_state.chat_history or 
             st.session_state.chat_history[-1]['question'] != user_question)):
            st.session_state.processing_query = True
            with st.spinner("Analyzing your question... Don’t expect me to go easy on you!"):
                try:
                    filters = {
                        'topic': st.session_state.analysis_results['metadata']['topic'],
                        'country': st.session_state.analysis_results['metadata'].get('country')
                    }
                    filters = {k: v for k, v in filters.items() if v is not None}
                    answer = st.session_state.pipeline.advisor.answer_custom_financial_query(
                        user_question, filters
                    )
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': answer,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
                    st.success("Response generated! Check the AI Chat tab to see me tear into your question.")
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
                finally:
                    st.session_state.processing_query = False
        elif user_question and not st.session_state.get('pipeline'):
            st.warning("Run an analysis first to enable the AI assistant! Don’t waste my time!")
        if st.session_state.chat_history:
            with st.expander("Recent Conversation", expanded=True):
                latest_chat = st.session_state.chat_history[-1]
                st.markdown(f"**You ({latest_chat['timestamp']}):** {latest_chat['question']}")
                st.markdown(f"**FinWise AI:** {latest_chat['answer']}")
                if len(st.session_state.chat_history) > 1:
                    st.info(f"{len(st.session_state.chat_history)} total conversations. View all in the AI Chat tab.")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>FinWise AI - Enhanced Global Financial Intelligence Platform</strong></p>
    <p>Analyze any topic, anywhere, anytime with AI-powered insights</p>
    <p><small>This tool provides informational analysis only. Not financial advice. Always consult with qualified professionals.</small></p>
    <p><strong>Created by Prince Patel</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_enhanced_streamlit_app()
