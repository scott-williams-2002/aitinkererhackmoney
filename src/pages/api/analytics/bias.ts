import type { APIRoute } from 'astro';
import { KalshiApiClient, DataProcessor } from '../utils/kalshi-api';
import type { AnalyticsResponse } from '../../types';

export const POST: APIRoute = async ({ request }) => {
  try {
    const { ticker } = await request.json();
    
    if (!ticker) {
      throw new Error('Ticker is required');
    }

    const kalshiClient = new KalshiApiClient(
      import.meta.env.KALSHI_API_KEY,
      import.meta.env.KALSHI_ENVIRONMENT as 'demo' | 'prod'
    );

    // Get market details
    const market = await kalshiClient.getMarket(ticker);
    const normalizedMarket = DataProcessor.normalizeMarket(market);

    // Get price history
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 30); // Last 30 days
    
    const priceHistory = await kalshiClient.getMarketPrices(ticker, startDate, endDate);

    // Calculate bias metrics
    const { BiasAnalytics } = await import('../utils/kalshi-api');
    
    const volatility = BiasAnalytics.calculateVolatility(priceHistory);
    const marketEfficiency = BiasAnalytics.calculateMarketEfficiency(
      priceHistory[priceHistory.length - 1]?.last_price || 50, 
      priceHistory
    );

    // TODO: Get category comparison data from Raindrop Smart SQL
    // const categoryMarkets = await raindrop.smartSQL.query(
    //   'SELECT * FROM market_prices_daily WHERE category = ?',
    //   [normalizedMarket.category]
    // );
    
    // For now, use mock category comparison
    const categoryComparison = Math.random() * 2 - 1; // -1 to 1

    const biasAnalysis = {
      ticker,
      analysis_date: new Date(),
      category: normalizedMarket.category,
      price_volatility: volatility,
      volume_bias_score: Math.random() * 2 - 1, // TODO: Calculate properly
      prediction_error: Math.random() * 10, // TODO: Calculate properly
      market_efficiency_score: marketEfficiency,
      confidence_level: 0.85,
      bias_direction: volatility > 0.3 ? 'high' : volatility < 0.1 ? 'low' : 'neutral' as 'high' | 'low' | 'neutral',
      category_avg_comparison: categoryComparison,
      analyzed_at: new Date()
    };

    // TODO: Store in Raindrop Smart SQL
    // await raindrop.smartSQL.upsert('bias_analysis', biasAnalysis);

    const response: AnalyticsResponse = {
      success: true,
      data: biasAnalysis
    };

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Bias analysis error:', error);
    
    const response: AnalyticsResponse = {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };

    return new Response(JSON.stringify(response), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};