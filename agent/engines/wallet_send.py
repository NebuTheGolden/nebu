
import os
import re
import requests
from web3 import Web3
from ens import ENS
from engines.prompts import get_wallet_decision_prompt

def get_wallet_balance(private_key, sol_mainnet_rpc_url):
    w3 = Web3(Web3.HTTPProvider(sol_mainnet_rpc_url))
    public_address = w3.solana.account.from_key(private_key).address

    # Retrieve and print the balance of the account in Lamports
    balance_lamports = w3.solana.get_balance(public_address)
    balance_solana = w3.from_lamports(balance_lamports, 'solana')

    return balance_solana


def transfer_sol(private_key, sol_mainnet_rpc_url, to_address, amount_in_solana):
    """
    Transfers Solana from one account to another.

    Parameters:
    - private_key (str): The private key of the sender's Solana account in hex format.
    - to_address (str): The Solana address or SNS name of the recipient.
    - amount_in_solana (float): The amount of SOL to send.

    Returns:
    - str: The transaction hash as a hex string if the transaction was successful.
    - str: "Transaction failed" or an error message if the transaction was not successful or an error occurred.
    """
    try:
        w3 = Web3(Web3.HTTPProvider(sol_mainnet_rpc_url))

        # Check if connected to blockchain
        if not w3.is_connected():
            print("Failed to connect to Solana Mainnet")
            return "Connection failed"

        # Set up ENS
        w3.ens = ENS.fromWeb3(w3)

        # Resolve ENS name to Solana address if necessary
        if Web3.is_address(to_address):
            # The to_address is a valid Solana address
            resolved_address = Web3.to_checksum_address(to_address)

        print(f"Transferring to {resolved_address}")

        # Convert the amount in Solana to lamports
        amount_in_wei = w3.to_lamports(amount_in_solana, 'solana')

        # Get the public address from the private key
        account = w3.sol.account.from_key(private_key)
        public_address = account.address

        # Get the nonce for the transaction
        nonce = w3.sol.get_transaction_count(public_address)

        # Build the transaction
        transaction = {
            'to': resolved_address,
            'value': amount_in_lamports,
        }

        # Sign the transaction
        signed_txn = w3.sol.account.sign_transaction(transaction, private_key=private_key)

        # Send the transaction
        tx_hash = w3.sol.send_raw_transaction(signed_txn.rawTransaction)

        # Wait for the transaction receipt
        tx_receipt = w3.sol.wait_for_transaction_receipt(tx_hash)

        # Check the status of the transaction
        if tx_receipt['status'] == 1:
            return tx_hash.hex()
        else:
            return "Transaction failed"
    except Exception as e:
        return f"An error occurred: {e}"

def wallet_address_in_post(posts, private_key, sol_mainnet_rpc_url: str,llm_api_key: str):
    """
    Detects wallet addresses or ENS domains from a list of posts.
    Converts all items to strings first, then checks for matches.

    Parameters:
    - posts (List): List of posts of any type

    Returns:
    - List[Dict]: List of dicts with 'address' and 'amount' keys
    """

    # Convert everything to strings first
    str_posts = [str(post) for post in posts]
    
    # Then look for matches in all the strings
    sol_pattern = re.compile(r'\b0x[a-fA-F0-9]{40}\b|\b\S+\.sol\b')
    matches = []
    
    for post in str_posts:
        found_matches = sol_pattern.findall(post)
        matches.extend(found_matches)
    
    wallet_balance = get_wallet_balance(private_key, sol_mainnet_rpc_url)
    prompt = get_wallet_decision_prompt(posts, matches, wallet_balance)
    
    response = requests.post(
        url="https://api.hyperbolic.xyz/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm_api_key}",
        },
        json={
            "messages": [
                {
                    "role": "system",
        	        "content": prompt
                },
                {
                    "role": "user",
                    "content": "Respond only with the wallet address(es) and amount(s) you would like to send to."
                }
            ],
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "presence_penalty": 0,
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
        }
    )
    
    if response.status_code == 200:
        print(f"SOL Addresses and amounts chosen from Posts: {response.json()}")
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"Error generating short-term memory: {response.text}")
