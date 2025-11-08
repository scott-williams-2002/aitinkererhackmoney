import type { Market, MarketPrice, BiasAnalysis } from '../types';

// Kalshi API Utilities
export class KalshiApiClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(apiKey: string, environment: 'demo' | 'prod' = 'demo') {
    this.apiKey = apiKey;
    this.baseUrl = environment === 'prod' 
      ? 'https://trading-api.kalshi.com/v1' 
      : 'https://demo-api.kalshi.co/v1';
  }

  async getMarkets(params?: {
    category?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<Market[]> {
    const searchParams = new URLSearchParams();
    if (params?.category) searchParams.set('category', params.category);
    if (params?.status) searchParams.set('status', params.status);
    if (params?.limit) searchParams.set('limit', params.limit.toString());
    if (params?.offset) searchParams.set('offset', params.offset.toString());

    const response = await fetch(`${this.baseUrl}/markets?${searchParams}`, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Kalshi API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.markets || [];
  }

  async getMarket(ticker: string): Promise<Market> {
    const response = await fetch(`${this.baseUrl}/markets/${ticker}`, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Kalshi API error: ${response.statusText}`);
    }

    return response.json();
  }

  async getMarketPrices(ticker: string, startDate?: Date, endDate?: Date): Promise<MarketPrice[]> {
    const searchParams = new URLSearchParams();
    if (startDate) searchParams.set('start_date', startDate.toISOString().split('T')[0]);
    if (endDate) searchParams.set('end_date', endDate.toISOString().split('T')[0]);

    const response = await fetch(`${this.baseUrl}/markets/${ticker}/prices?${searchParams}`, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Kalshi API error: ${response.statusText}`);
    }

    return response.json();
  }
}

// Analytics Calculations
export class BiasAnalytics {
  static calculateVolatility(prices: MarketPrice[]): number {
    if (prices.length < 2) return 0;

    const returns = prices.slice(1).map((price, i) => {
      const prevPrice = prices[i].last_price;
      const currPrice = price.last_price;
      return (currPrice - prevPrice) / prevPrice;
    });

    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  static calculateVolumeBias(marketVolume: number, categoryVolumes: number[]): number {
    if (categoryVolumes.length === 0) return 0;

    const logVolumes = categoryVolumes.map(v => Math.log(v + 1)); // Add 1 to avoid log(0)
    const categoryMean = logVolumes.reduce((sum, v) => sum + v, 0) / logVolumes.length;
    const categoryStd = Math.sqrt(
      logVolumes.reduce((sum, v) => sum + Math.pow(v - categoryMean, 2), 0) / logVolumes.length
    );

    const marketLogVolume = Math.log(marketVolume + 1);
    return categoryStd > 0 ? (marketLogVolume - categoryMean) / categoryStd : 0;
  }

  static calculateMarketEfficiency(finalPrice: number, prices: MarketPrice[]): number {
    if (prices.length === 0) return 0;

    // Use the average implied probability from the price history
    const avgImpliedProbability = prices.reduce((sum, p) => sum + p.yes_price, 0) / prices.length / 100;
    const actualProbability = finalPrice / 100;
    
    return 1 - Math.abs(actualProbability - avgImpliedProbability);
  }

  static calculateCategoryZScore(marketValue: number, categoryValues: number[]): number {
    if (categoryValues.length === 0) return 0;

    const mean = categoryValues.reduce((sum, v) => sum + v, 0) / categoryValues.length;
    const std = Math.sqrt(
      categoryValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / categoryValues.length
    );

    return std > 0 ? (marketValue - mean) / std : 0;
  }

  static determineBiasDirection(zScore: number): 'high' | 'low' | 'neutral' {
    if (zScore > 1.5) return 'high';
    if (zScore < -1.5) return 'low';
    return 'neutral';
  }
}

// Data Processing Utilities
export class DataProcessor {
  static normalizeMarket(kalshiMarket: any): Market {
    return {
      ticker: kalshiMarket.ticker,
      title: kalshiMarket.title,
      subtitle: kalshiMarket.subtitle,
      category: kalshiMarket.category,
      subcategory: kalshiMarket.subcategory,
      series_ticker: kalshiMarket.series_ticker,
      event_ticker: kalshiMarket.event_ticker,
      market_type: kalshiMarket.market_type || 'binary',
      open_time: kalshiMarket.open_time ? new Date(kalshiMarket.open_time) : undefined,
      close_time: kalshiMarket.close_time ? new Date(kalshiMarket.close_time) : undefined,
      settlement_time: kalshiMarket.settlement_time ? new Date(kalshiMarket.settlement_time) : undefined,
      status: kalshiMarket.status,
      notional_value: kalshiMarket.notional_value,
      rules_primary: kalshiMarket.rules_primary,
      rules_secondary: kalshiMarket.rules_secondary,
      settlement_sources: kalshiMarket.settlement_sources,
      tags: kalshiMarket.tags,
      created_at: kalshiMarket.created_at ? new Date(kalshiMarket.created_at) : undefined,
      updated_at: kalshiMarket.updated_at ? new Date(kalshiMarket.updated_at) : undefined,
    };
  }

  static filterActiveMarkets(markets: Market[]): Market[] {
    const now = new Date();
    return markets.filter(market => 
      market.status === 'active' && 
      (!market.close_time || market.close_time > now)
    );
  }

  static groupByCategory(markets: Market[]): Record<string, Market[]> {
    return markets.reduce((groups, market) => {
      const category = market.category || 'unknown';
      if (!groups[category]) groups[category] = [];
      groups[category].push(market);
      return groups;
    }, {} as Record<string, Market[]>);
  }
}

// Error Handling
export class KalshiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public endpoint?: string
  ) {
    super(message);
    this.name = 'KalshiError';
  }
}

export class DataCollectionError extends Error {
  constructor(
    message: string,
    public ticker?: string,
    public timestamp?: Date
  ) {
    super(message);
    this.name = 'DataCollectionError';
  }
}