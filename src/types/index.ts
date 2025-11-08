// Shared types for frontend/backend communication
export interface Market {
  ticker: string;
  title: string;
  subtitle?: string;
  category: string;
  subcategory?: string;
  series_ticker?: string;
  event_ticker?: string;
  market_type: 'binary' | 'other';
  open_time?: Date;
  close_time?: Date;
  settlement_time?: Date;
  status: 'active' | 'closed' | 'settled';
  notional_value?: number;
  rules_primary?: string;
  rules_secondary?: string;
  settlement_sources?: object;
  tags?: string[];
  created_at?: Date;
  updated_at?: Date;
}

export interface MarketPrice {
  ticker: string;
  price_date: Date;
  price_time: string;
  yes_price: number;
  no_price: number;
  yes_bid: number;
  yes_ask: number;
  no_bid: number;
  no_ask: number;
  last_price: number;
  volume: number;
  open_interest: number;
  liquidity: number;
  spread_cents: number;
}

export interface BiasAnalysis {
  ticker: string;
  analysis_date: Date;
  category: string;
  price_volatility: number;
  volume_bias_score: number;
  prediction_error: number;
  market_efficiency_score: number;
  confidence_level: number;
  bias_direction: 'high' | 'low' | 'neutral';
  category_avg_comparison: number;
  analyzed_at: Date;
}

// API Request/Response types
export interface MarketSearchParams {
  query?: string;
  category?: string;
  subcategory?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

export interface AnalyticsResponse {
  success: boolean;
  data?: BiasAnalysis;
  error?: string;
}

export interface MarketsResponse {
  success: boolean;
  data?: Market[];
  total?: number;
  error?: string;
}