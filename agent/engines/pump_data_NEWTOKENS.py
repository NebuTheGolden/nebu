import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import time
import logging
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.exceptions import TransportQueryError

@dataclass
class TokenData:
    """Data class to store token information"""
    mint_address: str
    owner_address: str
    program_id: str
    is_writable: bool

@dataclass
class ProgramData:
    """Data class to store program information"""
    address: str
    method: str
    name: str
    account_names: List[str]
    arguments: Dict[str, Union[str, int, float, bool]]

@dataclass
class InstructionData:
    """Data class to store instruction information"""
    tokens: List[TokenData]
    logs: List[str]
    program: ProgramData
    signature: str
    timestamp: datetime

class PumpFunAnalyzer:
    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://graphql.bitquery.io",
        cache_dir: str = "./pump_data_cache"
    ):
        """
        Initialize the PumpFun analyzer with Bitquery credentials
        
        Args:
            api_key: Bitquery API key
            endpoint: Bitquery GraphQL endpoint
            cache_dir: Directory to cache retrieved data
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

    async def fetch_new_tokens_query(self) -> str:
        """Returns the GraphQL query for new token creation"""
        return """
        subscription {
          Solana {
            Instructions(
              where: {
                Instruction: {
                  Program: { Method: { is: "create" }, Name: { is: "pump" } }
                }
              }
            ) {
              Instruction {
                Accounts {
                  Address
                  IsWritable
                  Token {
                    Mint
                    Owner
                    ProgramId
                  }
                }
                Logs
                Program {
                  AccountNames
                  Address
                  Arguments {
                    Name
                    Type
                    Value {
                      ... on Solana_ABI_Json_Value_Arg {
                        json
                      }
                      ... on Solana_ABI_Float_Value_Arg {
                        float
                      }
                      ... on Solana_ABI_Boolean_Value_Arg {
                        bool
                      }
                      ... on Solana_ABI_Bytes_Value_Arg {
                        hex
                      }
                      ... on Solana_ABI_BigInt_Value_Arg {
                        bigInteger
                      }
                      ... on Solana_ABI_Address_Value_Arg {
                        address
                      }
                      ... on Solana_ABI_String_Value_Arg {
                        string
                      }
                      ... on Solana_ABI_Integer_Value_Arg {
                        integer
                      }
                    }
                  }
                  Method
                  Name
                }
              }
              Transaction {
                Signature
              }
            }
          }
        }
        """

    def _parse_token_data(self, account_data: Dict) -> TokenData:
        """Parse raw account data into TokenData object"""
        return TokenData(
            mint_address=account_data.get('Token', {}).get('Mint', ''),
            owner_address=account_data.get('Token', {}).get('Owner', ''),
            program_id=account_data.get('Token', {}).get('ProgramId', ''),
            is_writable=account_data.get('IsWritable', False)
        )

    def _parse_program_data(self, program_data: Dict) -> ProgramData:
        """Parse raw program data into ProgramData object"""
        arguments = {}
        for arg in program_data.get('Arguments', []):
            name = arg.get('Name', '')
            value = arg.get('Value', {})
            
            # Handle different value types
            for value_type in ['json', 'float', 'bool', 'hex', 'bigInteger', 
                             'address', 'string', 'integer']:
                if value_type in value:
                    arguments[name] = value[value_type]
                    break
        
        return ProgramData(
            address=program_data.get('Address', ''),
            method=program_data.get('Method', ''),
            name=program_data.get('Name', ''),
            account_names=program_data.get('AccountNames', []),
            arguments=arguments
        )

    async def fetch_pump_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[InstructionData]:
        """
        Fetch PumpFun data from Bitquery
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            
        Returns:
            List of InstructionData objects
        """
        query = await self.fetch_new_tokens_query()
        
        try:
            # Execute the query
            result = await self.client.execute_async(gql(query))
            
            instructions = []
            for instruction_data in result['Solana']['Instructions']:
                # Parse instruction data
                instruction = instruction_data['Instruction']
                
                # Parse tokens
                tokens = [
                    self._parse_token_data(account)
                    for account in instruction.get('Accounts', [])
                ]
                
                # Parse program data
                program = self._parse_program_data(instruction['Program'])
                
                # Create instruction object
                instruction_obj = InstructionData(
                    tokens=tokens,
                    logs=instruction.get('Logs', []),
                    program=program,
                    signature=instruction_data['Transaction']['Signature'],
                    timestamp=datetime.now()  # You might want to get this from the blockchain
                )
                
                instructions.append(instruction_obj)
            
            return instructions
            
        except TransportQueryError as e:
            self.logger.error(f"Query error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise

    def analyze_pump_activity(
        self,
        instructions: List[InstructionData]
    ) -> pd.DataFrame:
        """
        Analyze PumpFun activity data
        
        Args:
            instructions: List of instruction data
            
        Returns:
            DataFrame with analyzed data
        """
        analyzed_data = []
        
        for inst in instructions:
            # Extract relevant data for analysis
            analysis = {
                'timestamp': inst.timestamp,
                'signature': inst.signature,
                'program_name': inst.program.name,
                'method': inst.program.method,
                'num_tokens': len(inst.tokens),
                'num_writable_tokens': sum(1 for t in inst.tokens if t.is_writable),
                'has_logs': len(inst.logs) > 0
            }
            
            # Add program arguments
            for arg_name, arg_value in inst.program.arguments.items():
                analysis[f'arg_{arg_name}'] = arg_value
            
            analyzed_data.append(analysis)
        
        return pd.DataFrame(analyzed_data)

    async def monitor_pump_activity(
        self,
        interval: int = 60,
        output_dir: Optional[Path] = None
    ):
        """
        Continuously monitor PumpFun activity
        
        Args:
            interval: Polling interval in seconds
            output_dir: Directory to save monitoring results
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        while True:
            try:
                # Fetch new data
                instructions = await self.fetch_pump_data()
                
                if instructions:
                    # Analyze data
                    analysis_df = self.analyze_pump_activity(instructions)
                    
                    # Save results if output directory specified
                    if output_dir:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        analysis_df.to_csv(
                            output_dir / f"pump_analysis_{timestamp}.csv",
                            index=False
                        )
                    
                    # Log summary
                    self.logger.info(
                        f"Processed {len(instructions)} instructions. "
                        f"Found {analysis_df['num_tokens'].sum()} total tokens."
                    )
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Short delay before retry

# Example usage
async def main():
    # Initialize analyzer
    analyzer = PumpFunAnalyzer(
        api_key="YOUR_BITQUERY_API_KEY",
        cache_dir="./pump_data"
    )
    
    try:
        # Fetch recent data
        instructions = await analyzer.fetch_pump_data(
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        # Analyze data
        analysis_df = analyzer.analyze_pump_activity(instructions)
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total instructions: {len(instructions)}")
        print(f"Total tokens: {analysis_df['num_tokens'].sum()}")
        print("\nRecent activity:")
        print(analysis_df.head())
        
        # Start monitoring
        await analyzer.monitor_pump_activity(
            interval=60,
            output_dir=Path("./pump_monitoring")
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
