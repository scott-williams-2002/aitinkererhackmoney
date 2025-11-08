# Kalshi Market Bias Analytics Platform

A comprehensive platform for collecting, storing, and analyzing Kalshi prediction market data to identify and measure pricing bias across different market categories.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit      │    │   Netlify Edge   │    │   Raindrop AI   │    │   Kalshi API    │
│   Frontend       │◄──►│   Backend API    │◄──►│   Smart SQL      │◄──►│   Data Source    │
│                 │    │                  │    │   Time Series   │    │                 │
│ • Market Search │    │ • Auth & Rate    │    │ • Market Store  │    │ • Market Data   │
│ • Bias Analysis │    │ • Data Pipeline  │    │ • Analytics     │    │ • Historical     │
│ • Charts/Viz    │    │ • AI Agent       │    │ • Pattern Query  │    │ • Rules/Events   │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### Data Collection
- **Market Discovery**: Daily batch runs for all active and recently closed markets
- **Price Data**: Daily collection at noon (12:00 UTC) for all markets with trading activity
- **Event Tracking**: Real-time and historical event data, settlements, and rule changes

### Analytics & Bias Detection
- **Volatility Analysis**: Standard deviation of daily price changes
- **Volume Bias Score**: Volume deviation from category average
- **Market Efficiency**: Price prediction accuracy vs final outcome
- **Category Comparison**: Z-score relative to category peers

### AI-Powered Features
- **Smart Search**: Natural language market discovery using semantic search
- **Pattern Detection**: AI-powered bias pattern identification
- **Predictive Analytics**: Market efficiency and bias trend analysis

## Technology Stack

- **Frontend**: Astro + React + Tailwind CSS
- **Backend**: Netlify Edge Functions
- **AI Infrastructure**: Raindrop (Smart SQL, Smart Memory, Smart Search)
- **Data Source**: Kalshi API
- **Deployment**: Netlify

## Quick Start

### Prerequisites
- [Node.js](https://nodejs.org/) v18.14+
- [Netlify CLI](https://docs.netlify.com/cli/get-started/)
- [Raindrop account](https://liquidmetal.ai) with API key
- [Kalshi API](https://kalshi.com/) credentials

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd kalshi-analytics-platform
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Start development server:
```bash
npm run dev
```

### Environment Variables

Required environment variables:

```bash
# Raindrop Configuration
RAINDROP_API_KEY=your_raindrop_api_key
RAINDROP_SMARTSQL_NAME=your_smartsql_instance
RAINDROP_SMARTMEMORY_NAME=your_smartmemory_instance
RAINDROP_APPLICATION_NAME=kalshi-analytics
RAINDROP_APPLICATION_VERSION=1.0.0

# Kalshi API
KALSHI_API_KEY=your_kalshi_api_key
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
KALSHI_ENVIRONMENT=prod  # or 'demo' for testing

# Netlify
NETLIFY_SITE_ID=your_netlify_site_id
```

## Project Structure

```
src/
├── pages/
│   ├── dashboard/           # Main dashboard and overview
│   ├── markets/            # Market search and details
│   ├── analytics/          # Bias analysis and visualizations
│   └── api/                # Backend API endpoints
│       ├── markets/        # Market data endpoints
│       ├── analytics/      # Analytics endpoints
│       ├── data/          # Data collection endpoints
│       └── collection/    # Collection automation
├── components/            # Reusable React components
├── utils/                # Utility functions
└── types/                # TypeScript type definitions
```

## API Endpoints

### Market Data
- `GET /api/markets/search` - Search markets with filters
- `GET /api/markets/{ticker}` - Get market details
- `GET /api/series/{ticker}` - Get series information

### Analytics
- `GET /api/analytics/bias/{ticker}` - Get bias analysis for a market
- `GET /api/analytics/category-bias/{category}` - Category-level bias analysis
- `POST /api/analytics/calculate-batch` - Batch analysis calculations

### Data Collection
- `POST /api/data/collect/daily` - Trigger daily price collection
- `POST /api/data/backfill/{range}` - Backfill historical data
- `GET /api/data/collection/status` - Collection progress status

## Data Schema

### Core Tables

#### Markets
- Market metadata, rules, settlement sources
- Categories, subcategories, series information
- Open/close/settlement times

#### Daily Prices
- Yes/No prices, volume, bid/ask spreads
- Time series data at daily intervals
- Liquidity and spread metrics

#### Bias Analysis
- Volatility metrics, volume bias scores
- Prediction error calculations
- Market efficiency scores

## Development

### Running Locally
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
```

### Database Schema
The platform uses Raindrop Smart SQL for data storage. See `architecture-plan.md` for the complete schema definition.

### Data Collection
Data collection runs on automated schedules:
- **Market Discovery**: Daily at 00:00 UTC
- **Price Collection**: Daily at 12:00 UTC
- **Analysis Updates**: Hourly or on-demand

## Deployment

### Deploy to Netlify
1. Connect your repository to Netlify
2. Set up environment variables in Netlify dashboard
3. Configure build settings:
   - Build command: `npm run build`
   - Publish directory: `dist`
4. Deploy!

## Monitoring & Maintenance

### Data Quality
- Collection success rate monitoring
- Data freshness checks
- Coverage metrics tracking
- Validation against known outcomes

### Performance
- API response time monitoring
- Database query optimization
- Edge function caching strategies
- Batch processing optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- [Raindrop Documentation](https://docs.liquidmetal.ai)
- [Kalshi API Documentation](https://trading-api.kalshi.com/v1/docs)
- [Netlify Documentation](https://docs.netlify.com)

---

*Built with ❤️ using Raindrop AI, Netlify, and Kalshi APIs*