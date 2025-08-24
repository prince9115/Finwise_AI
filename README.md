# 🚀 FinWise AI Enhanced

**AI-Powered Financial Intelligence Platform for Public Use**

A comprehensive financial analysis tool that allows users to analyze any financial topic with their own API keys. Perfect for students, researchers, individual investors, and finance professionals.

## ✨ Key Features

### 🔐 **User-Provided API Keys**
- Secure API key validation before analysis
- Clear setup instructions for all required services
- All APIs offer generous free tiers perfect for testing

### 🎮 **Demo Mode**
- Try the platform without API keys
- Pre-loaded sample data and analysis
- Perfect for exploring features before setup

### 🧠 **Enhanced AI Analysis**
- Smart topic sentiment analysis using Groq LLM
- Market trend analysis with context
- Sarcastic but educational financial advisor persona
- Vector-based document search with Cohere embeddings

### 📊 **Comprehensive Data Sources**
- **NewsAPI**: Global news coverage (40+ countries)
- **Alpha Vantage**: Stock market data and financial indicators
- **Multiple export formats**: PDF reports, JSON, CSV

### 🚨 **Production-Ready Features**
- Rate limiting and API quota management
- Enhanced error handling with user-friendly messages
- Progress tracking and status updates
- Comprehensive input validation

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements_enhanced.txt
```

### 3. Get Your Free API Keys

**NewsAPI** (Free: 1000 requests/day)
- Visit: [newsapi.org](https://newsapi.org)
- Sign up and get your free API key

**Alpha Vantage** (Free: 5 calls/min, 500/day)  
- Visit: [alphavantage.co](https://www.alphavantage.co)
- Get free API key for stock data

**Cohere** (Free: 100 calls/month)
- Visit: [cohere.ai](https://cohere.ai) 
- Required for AI embeddings and search

**Groq** (Free tier available)
- Visit: [groq.com](https://groq.com)
- Fastest AI inference for chat features

### 4. Run the Application
```bash
streamlit run finwise_ai_enhanced.py
```

## 🎯 How to Use

### **Quick Start (Demo Mode)**
1. ✅ Check "Demo Mode" in the sidebar
2. 📊 Explore pre-loaded AI analysis results
3. 💬 Try the AI chat feature with sample data

### **Full Analysis (With Your API Keys)**
1. 🔑 Enter your API keys in the sidebar
2. ✅ Click "Test API Keys" to validate
3. 📝 Choose a topic or use quick presets
4. 🚀 Click "Run Enhanced Analysis"
5. 📊 Explore results in organized tabs

## 🎲 Sample Analysis Topics

- **AI & Technology**: "artificial intelligence machine learning"
- **Renewable Energy**: "solar wind renewable energy climate"  
- **Healthcare**: "biotechnology pharmaceuticals healthcare innovation"
- **Cryptocurrency**: "bitcoin ethereum cryptocurrency blockchain"
- **ESG Investing**: "ESG sustainability climate change environment"

## 📈 What You Get

### **Comprehensive Analysis Reports**
- 📰 News sentiment analysis with market implications
- 📊 Technical and fundamental market analysis  
- 📋 Executive summary with key metrics
- 📉 Interactive charts and visualizations

### **AI Financial Advisor**
- 🤖 Context-aware responses using your analysis
- 📚 Educational explanations with analogies
- ⚠️ Risk assessment and opportunity identification
- 💡 Actionable insights when data supports it

### **Export Capabilities**
- 📄 Professional PDF reports
- 📊 CSV data for further analysis
- 💾 JSON data for developers

## ⚡ Performance & Limits

### **Free Tier Capabilities**
- **NewsAPI**: Up to 1000 articles/day
- **Alpha Vantage**: 500 market data points/day
- **Cohere**: 100 AI embedding calls/month  
- **Groq**: Generous free tier for chat

### **Optimizations**
- ✅ Intelligent rate limiting
- ✅ Progress tracking for long operations
- ✅ Batch processing for efficiency
- ✅ Error recovery and user guidance

## 🔒 Privacy & Security

- 🔐 API keys are never stored permanently
- 🚫 No user data collection or tracking
- 🛡️ Secure session-based storage only
- ⚠️ Users manage their own API quotas

## 💡 Educational Use Cases

### **For Students & Researchers**
- 📖 Learn financial analysis concepts
- 🧪 Experiment with different topics
- 📊 Practice data interpretation skills
- 💼 Build portfolio analysis experience

### **For Professionals**  
- 🔍 Quick market research and sentiment analysis
- 📈 Competitive intelligence gathering
- 📋 Report generation for presentations
- 🎯 Topic-specific market insights

## 🐛 Troubleshooting

### **API Key Issues**
```
❌ Invalid API key
→ Double-check key from provider website
→ Ensure no extra spaces or characters

⏰ Rate limit exceeded  
→ Wait for quota reset (usually daily)
→ Consider upgrading to paid tier
```

### **Common Errors**
```
🌐 Connection failed
→ Check internet connection
→ Verify API service status

📊 No data returned
→ Try broader search terms
→ Check date range settings
→ Verify stock symbols exist
```

## 🚀 Deployment Options

### **Streamlit Cloud** (Recommended)
1. Fork this repository
2. Connect to Streamlit Cloud
3. Users enter their own API keys
4. Perfect for public sharing

### **Heroku/Railway** 
- Set environment variables for any default configs
- Users still provide their own API keys
- Good for custom domain needs

### **Local Sharing**
- Run locally and share via ngrok
- Good for team/classroom use

## 🤝 Contributing

This project is designed for educational and public use. Contributions welcome for:

- 🐛 Bug fixes and improvements
- 📚 Documentation updates
- ✨ New analysis features
- 🎨 UI/UX enhancements

## ⚠️ Important Disclaimers

**Financial Advice**: This tool provides educational analysis only and is not financial advice. All analysis is based on publicly available data and AI interpretation. Always consult qualified financial professionals before making investment decisions.

**API Costs**: While all APIs offer free tiers, users are responsible for managing their own usage and any costs beyond free limits.

**Data Accuracy**: Analysis depends on third-party data sources. Always verify important information independently.

## 📞 Support

- 📖 Check the in-app help section
- 🔗 Review API provider documentation  
- 🐛 Report issues via GitHub Issues
- 💡 Feature requests welcome

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for the financial education community**

*Empowering users with AI-driven financial insights while maintaining complete control over their data and API usage.*