import type { APIRoute } from 'astro';
import { KalshiApiClient, DataProcessor } from '../utils/kalshi-api';
import type { MarketSearchParams, MarketsResponse } from '../../types';

export const POST: APIRoute = async ({ request }) => {
  try {
    const searchParams: MarketSearchParams = await request.json();
    
    const kalshiClient = new KalshiApiClient(
      import.meta.env.KALSHI_API_KEY,
      import.meta.env.KALSHI_ENVIRONMENT as 'demo' | 'prod'
    );

    // Get markets from Kalshi
    const kalshiMarkets = await kalshiClient.getMarkets({
      category: searchParams.category,
      status: searchParams.status || 'active',
      limit: searchParams.limit || 50,
      offset: searchParams.offset || 0
    });

    // Normalize and process the data
    const markets = kalshiMarkets.map(market => DataProcessor.normalizeMarket(market));

    // TODO: Store in Raindrop Smart SQL for future queries
    // await raindrop.smartSQL.upsertMany('markets', markets);

    const response: MarketsResponse = {
      success: true,
      data: markets,
      total: markets.length
    };

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Markets search error:', error);
    
    const response: MarketsResponse = {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };

    return new Response(JSON.stringify(response), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};