import asyncio
import aiohttp
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import time
import logging
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.exceptions import TransportQueryError
import json
from decimal import Decimal

@dataclass
class PoolToken:
    """Data class for token information in a pool"""
    mint_address: str
    symbol: str
    decimals: int = 9  # Default for Solana

@dataclass
class PoolMarket:
    """Data class for market information"""
    market_address: str
    token_a: PoolToken
    token_b: PoolToken
    trade_count: int
    last_trade_time: datetime
    volume_24h: Decimal
    liquidity: Decimal

class PumpPoolAnalyzer:
    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://graphql.bitquery.io",
        cache_dir: str = "./pump_pool_cache"
    ):
        """
        Initialize the PumpFun pool analyzer
        
        Args:
            api_key: Bitquery API key
            endpoint: Bitquery GraphQL endpoint
            cache_dir: Directory to cache data
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize GraphQL client
        transport = AIOHTTPTransport(
            url=endpoint,
            headers={'X-API-KEY': api_key}
        )
        self.client = Client(
            transport=transport,
            fetch_schema_from_transport=True
        )
        
        # Cache for market data
        self.market_cache: Dict[str, PoolMarket] = {}

    def get_pool_query(self, mint_address: Optional[str] = None) -> str:
        """
        Generate GraphQL query for pool details
        
        Args:
            mint_address: Optional specific mint address to query
        
        Returns:
            str: GraphQL query
        """
        where_clause = """
            Trade: {
                Dex: {ProtocolName: {is: "pump"}},
                Currency: {Symbol: {not: ""}}
            }
        """
        
        if mint_address:
            where_clause = f"""
                Trade: {{
                    Dex: {{ProtocolName: {{is: "pump"}}}},
                    Currency: {{
                        Symbol: {{not: ""}},
                        MintAddress: {{in: "{mint_address}"}}
                    }}
                }}
            """
        
        return f"""
        {{
          Solana {{
            DEXTradeByTokens(
              where: {{{where_clause}}}
            ) {{
              count
              Trade {{
                Market {{
                  MarketAddress
                  BaseToken {{
                    MintAddress
                    Symbol
                  }}
                  QuoteToken {{
                    MintAddress
                    Symbol
                  }}
                }}
                Block {{
                  Timestamp
                }}
                Currency {{
                  MintAddress
                  Symbol
                }}
                Volume {{
                  Amount
                  Currency {{
                    Symbol
                  }}
                }}
              }}
            }}
          }}
        }}
        """

    def get_market_stats_query(self, market_address: str) -> str:
        """Generate GraphQL query for market statistics"""
        return f"""
        {{
          Solana {{
            DEXTrades(
              where: {{
                Trade: {{
                  Market: {{Address: {{is: "{market_address}"}}}},
                  Dex: {{ProtocolName: {{is: "pump"}}}}
                }}
              }}
              options: {{limit: 100}}
            ) {{
              Volume {{
                Amount
                Currency {{
                  Symbol
                }}
              }}
              Price {{
                Amount
              }}
              Block {{
                Timestamp
              }}
            }}
          }}
        }}
        """

    async def fetch_pool_data(
        self,
        mint_address: Optional[str] = None
    ) -> List[PoolMarket]:
        """
        Fetch pool data from Bitquery
        
        Args:
            mint_address: Optional specific mint address to query
            
        Returns:
            List of PoolMarket objects
        """
        query = self.get_pool_query(mint_address)
        
        try:
            result = await self.client.execute_async(gql(query))
            trades = result['Solana']['DEXTradeByTokens']
            
            markets: Dict[str, PoolMarket] = {}
            
            for trade_data in trades:
                trade = trade_data['Trade']
                market = trade['Market']
                
                market_address = market['MarketAddress']
                
                if market_address not in markets:
                    markets[market_address] = PoolMarket(
                        market_address=market_address,
                        token_a=PoolToken(
                            mint_address=market['BaseToken']['MintAddress'],
                            symbol=market['BaseToken']['Symbol']
                        ),
                        token_b=PoolToken(
                            mint_address=market['QuoteToken']['MintAddress'],
                            symbol=market['QuoteToken']['Symbol']
                        ),
                        trade_count=0,
                        last_trade_time=datetime.fromtimestamp(0),
                        volume_24h=Decimal('0'),
                        liquidity=Decimal('0')
                    )
                
                # Update market statistics
                markets[market_address].trade_count += 1
                trade_time = datetime.fromisoformat(trade['Block']['Timestamp'].replace('Z', '+00:00'))
                markets[market_address].last_trade_time = max(
                    markets[market_address].last_trade_time,
                    trade_time
                )
                
                # Update volume if trade is within last 24 hours
                if datetime.now() - trade_time <= timedelta(days=1):
                    volume = Decimal(str(trade['Volume']['Amount']))
                    markets[market_address].volume_24h += volume
            
            return list(markets.values())
            
        except Exception as e:
            self.logger.error(f"Error fetching pool data: {str(e)}")
            raise

    async def fetch_market_details(self, market_address: str) -> Dict:
        """Fetch detailed market statistics"""
        query = self.get_market_stats_query(market_address)
        
        try:
            result = await self.client.execute_async(gql(query))
            return result['Solana']['DEXTrades']
        except Exception as e:
            self.logger.error(f"Error fetching market details: {str(e)}")
            raise

    def analyze_market_metrics(
        self,
        trades: List[Dict]
    ) -> Dict:
        """
        Calculate market metrics from trade data
        
        Args:
            trades: List of trade data
            
        Returns:
            Dict containing calculated metrics
        """
        if not trades:
            return {}
            
        volumes = [Decimal(str(t['Volume']['Amount'])) for t in trades]
        prices = [Decimal(str(t['Price']['Amount'])) for t in trades]
        timestamps = [
            datetime.fromisoformat(t['Block']['Timestamp'].replace('Z', '+00:00'))
            for t in trades
        ]
        
        # Calculate metrics
        metrics = {
            'volume_total': sum(volumes),
            'price_latest': prices[-1] if prices else Decimal('0'),
            'price_high_24h': max(prices),
            'price_low_24h': min(prices),
            'price_change_24h': (
                ((prices[-1] - prices[0]) / prices[0] * 100)
                if prices and prices[0] != 0
                else Decimal('0')
            ),
            'trade_count': len(trades),
            'first_trade_time': min(timestamps),
            'last_trade_time': max(timestamps)
        }
        
        return metrics

    async def monitor_pools(
        self,
        interval: int = 300,
        output_dir: Optional[Path] = None
    ):
        """
        Continuously monitor pool activity
        
        Args:
            interval: Polling interval in seconds
            output_dir: Directory to save monitoring results
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        while True:
            try:
                # Fetch all pools
                pools = await self.fetch_pool_data()
                
                pool_data = []
                for pool in pools:
                    # Fetch detailed market data
                    market_details = await self.fetch_market_details(pool.market_address)
                    metrics = self.analyze_market_metrics(market_details)
                    
                    pool_info = {
                        'market_address': pool.market_address,
                        'token_a_symbol': pool.token_a.symbol,
                        'token_a_mint': pool.token_a.mint_address,
                        'token_b_symbol': pool.token_b.symbol,
                        'token_b_mint': pool.token_b.mint_address,
                        'trade_count': pool.trade_count,
                        'volume_24h': float(pool.volume_24h),
                        'last_trade_time': pool.last_trade_time.isoformat(),
                        **{f'metric_{k}': float(v) if isinstance(v, Decimal) else v 
                           for k, v in metrics.items()}
                    }
                    
                    pool_data.append(pool_info)
                
                # Create DataFrame and save if output directory specified
                if pool_data and output_dir:
                    df = pd.DataFrame(pool_data)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    df.to_csv(
                        output_dir / f"pool_analysis_{timestamp}.csv",
                        index=False
                    )
                    
                    # Save detailed JSON
                    with open(output_dir / f"pool_details_{timestamp}.json", 'w') as f:
                        json.dump(pool_data, f, indent=2, default=str)
                    
                    self.logger.info(
                        f"Processed {len(pools)} pools. "
                        f"Total 24h volume: {sum(p['volume_24h'] for p in pool_data):,.2f}"
                    )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)

# Example usage
async def main():
    analyzer = PumpPoolAnalyzer(
        api_key="YOUR_BITQUERY_API_KEY",
        cache_dir="./pump_pool_data"
    )
    
    try:
        # Initial pool analysis
        pools = await analyzer.fetch_pool_data()
        
        print("\nPool Analysis Summary:")
        print(f"Total pools found: {len(pools)}")
        
        for pool in pools:
            print(f"\nMarket: {pool.market_address}")
            print(f"Pair: {pool.token_a.symbol}/{pool.token_b.symbol}")
            print(f"24h Volume: {float(pool.volume_24h):,.2f}")
            print(f"Trade Count: {pool.trade_count}")
        
        # Start continuous monitoring
        await analyzer.monitor_pools(
            interval=300,  # 5 minutes
            output_dir=Path("./pool_monitoring")
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
