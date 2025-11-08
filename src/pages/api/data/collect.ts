import type { APIRoute } from 'astro';
import { KalshiApiClient, DataProcessor } from '../utils/kalshi-api';

export const POST: APIRoute = async ({ request }) => {
  try {
    const { type = 'daily' } = await request.json();
    
    const kalshiClient = new KalshiApiClient(
      import.meta.env.KALSHI_API_KEY,
      import.meta.env.KALSHI_ENVIRONMENT as 'demo' | 'prod'
    );

    // Get all active markets
    const allMarkets = await kalshiClient.getMarkets({ status: 'active' });
    const activeMarkets = DataProcessor.filterActiveMarkets(allMarkets.map(market => 
      DataProcessor.normalizeMarket(market)
    ));

    const collectionResults = {
      total_markets: activeMarkets.length,
      collected: 0,
      errors: [],
      started_at: new Date()
    };

    // Collect price data for each active market
    for (const market of activeMarkets) {
      try {
        const priceData = await kalshiClient.getMarketPrices(market.ticker);
        
        // TODO: Store in Raindrop Smart SQL
        // await raindrop.smartSQL.upsertMany('market_prices_daily', priceData.map(price => ({
        //   ...price,
        //   price_date: price.price_date || new Date(),
        //   created_at: new Date()
        // })));

        collectionResults.collected++;
        
      } catch (error) {
        collectionResults.errors.push({
          ticker: market.ticker,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
        console.error(`Failed to collect price data for ${market.ticker}:`, error);
      }
    }

    collectionResults['completed_at'] = new Date();
    collectionResults['success_rate'] = collectionResults.collected / collectionResults.total_markets;

    return new Response(JSON.stringify({
      success: true,
      data: collectionResults
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Data collection error:', error);
    
    return new Response(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};