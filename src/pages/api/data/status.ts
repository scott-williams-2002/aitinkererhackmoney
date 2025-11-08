import type { APIRoute } from 'astro';

export const GET: APIRoute = async () => {
  try {
    // TODO: Get actual collection status from Raindrop Smart SQL
    // const lastCollection = await raindrop.smartSQL.queryOne(
    //   'SELECT * FROM collection_log ORDER BY completed_at DESC LIMIT 1'
    // );
    
    // Mock data for now
    const status = {
      last_collection: new Date(Date.now() - 1000 * 60 * 60 * 24), // 24 hours ago
      success_rate: 0.95,
      total_markets: 1250,
      collected_markets: 1187,
      errors: [
        'Failed to fetch data for INX-24DEC25-A-10',
        'Rate limit exceeded for some markets'
      ],
      next_scheduled: new Date(Date.now() + 1000 * 60 * 60 * 12), // 12 hours from now
      is_running: false
    };

    return new Response(JSON.stringify({
      success: true,
      data: status
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('Collection status error:', error);
    
    return new Response(JSON.stringify({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};