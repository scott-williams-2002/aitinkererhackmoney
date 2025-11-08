import { useState, useEffect } from 'react';
import type { Market, BiasAnalysis } from '../../types';

interface DashboardProps {
  className?: string;
}

export default function Dashboard({ className = '' }: DashboardProps) {
  const [markets, setMarkets] = useState<Market[]>([]);
  const [selectedMarket, setSelectedMarket] = useState<Market | null>(null);
  const [biasAnalysis, setBiasAnalysis] = useState<BiasAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMarkets = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/markets/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: 20, status: 'active' })
      });
      
      const result = await response.json();
      
      if (result.success) {
        setMarkets(result.data || []);
      } else {
        setError(result.error || 'Failed to fetch markets');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    } finally {
      setLoading(false);
    }
  };

  const fetchBiasAnalysis = async (ticker: string) => {
    try {
      const response = await fetch('/api/analytics/bias', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker })
      });
      
      const result = await response.json();
      
      if (result.success) {
        setBiasAnalysis(result.data);
      } else {
        setError(result.error || 'Failed to fetch bias analysis');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    }
  };

  const triggerDataCollection = async () => {
    try {
      const response = await fetch('/api/data/collect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'daily' })
      });
      
      const result = await response.json();
      
      if (result.success) {
        alert(`Data collection completed: ${result.data.collected}/${result.data.total_markets} markets`);
      } else {
        setError(result.error || 'Failed to collect data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error');
    }
  };

  useEffect(() => {
    fetchMarkets();
  }, []);

  useEffect(() => {
    if (selectedMarket) {
      fetchBiasAnalysis(selectedMarket.ticker);
    }
  }, [selectedMarket]);

  return (
    <div className={`p-6 ${className}`}>
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Kalshi Market Analytics</h1>
          <p className="text-gray-600">Monitor prediction market bias and analytics</p>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}

        <div className="mb-6">
          <button
            onClick={triggerDataCollection}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 mr-4"
          >
            Collect Data
          </button>
          <button
            onClick={fetchMarkets}
            disabled={loading}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Refresh Markets'}
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Markets List */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Active Markets</h2>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {markets.map((market) => (
                <div
                  key={market.ticker}
                  onClick={() => setSelectedMarket(market)}
                  className={`p-3 border rounded cursor-pointer hover:bg-gray-50 ${
                    selectedMarket?.ticker === market.ticker ? 'bg-blue-50 border-blue-300' : ''
                  }`}
                >
                  <div className="font-medium">{market.title}</div>
                  <div className="text-sm text-gray-600">
                    {market.ticker} • {market.category}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Bias Analysis */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Bias Analysis</h2>
            {selectedMarket && biasAnalysis ? (
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium">{selectedMarket.title}</h3>
                  <p className="text-sm text-gray-600">{selectedMarket.ticker}</p>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-600">Volatility</div>
                    <div className="text-lg font-semibold">{(biasAnalysis.price_volatility * 100).toFixed(2)}%</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Market Efficiency</div>
                    <div className="text-lg font-semibold">{(biasAnalysis.market_efficiency_score * 100).toFixed(2)}%</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Bias Direction</div>
                    <div className="text-lg font-semibold capitalize">{biasAnalysis.bias_direction}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Confidence</div>
                    <div className="text-lg font-semibold">{(biasAnalysis.confidence_level * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-500">Select a market to view bias analysis</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}