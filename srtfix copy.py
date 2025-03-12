import argparse
import re
import time
import os
from typing import List, Dict, Tuple
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

# Configuration constants
API_KEY = "AIzaSyCztyJmLgpfj9ko5G5Y48AxnzP2Nj78xYM"  # Replace with your actual API key
MODEL = "gemini-1.5-flash-8b"
BATCH_SIZE = 5  # Number of subtitle blocks to process in each API call
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 2  # Delay between retries in seconds

class SRTBlock:
    """Represents a single subtitle block in an SRT file."""
    
    def __init__(self, index: int, timestamp: str, text: str):
        self.index = index
        self.timestamp = timestamp
        self.text = text
    
    def __str__(self) -> str:
        return f"{self.index}\n{self.timestamp}\n{self.text}"

def setup_gemini_api():
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=API_KEY)

def parse_srt_file(file_path: str) -> List[SRTBlock]:
    """Parse an SRT file into a list of SRTBlock objects."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newline to get individual subtitle blocks
    raw_blocks = re.split(r'\n\n+', content.strip())
    srt_blocks = []
    
    for block in raw_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:  # Skip malformed blocks
            continue
        
        try:
            index = int(lines[0])
            timestamp = lines[1]
            text = '\n'.join(lines[2:])
            srt_blocks.append(SRTBlock(index, timestamp, text))
        except ValueError:
            # Skip blocks with non-integer indices
            continue
    
    return srt_blocks

def create_batches(srt_blocks: List[SRTBlock], batch_size: int) -> List[List[SRTBlock]]:
    """Divide the SRT blocks into batches of the specified size."""
    return [srt_blocks[i:i + batch_size] for i in range(0, len(srt_blocks), batch_size)]

def generate_correction_prompt(batch: List[SRTBlock]) -> str:
    """Generate a prompt for the Gemini API to correct a batch of subtitles."""
    prompt = """
以下は日本語の字幕（SRTファイル）の一部です。以下の問題を修正してください：
1. 文法エラーの修正
2. 明らかな誤字・脱字の修正
3. 繰り返しフレーズの削除
4. 途中で切れている文章の自然な補完
5. 不自然な表現の修正

元の字幕のタイムスタンプと番号はそのまま保持し、テキスト部分のみを修正してください。
修正した字幕は元の形式で返してください（番号、タイムスタンプ、テキストの順）。
極端に長い繰り返しがある場合は、それを短く自然にしてください。

【字幕データ】
"""
    
    for block in batch:
        prompt += f"{block}\n\n"
    
    prompt += """
【修正した字幕】
"""
    
    return prompt

def call_gemini_api(prompt: str) -> str:
    """Call the Gemini API with the given prompt and return the response."""
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            return response.text
        except (ResourceExhausted, ServiceUnavailable) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"API error: {e}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed after {MAX_RETRIES} attempts: {e}")
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

def parse_gemini_response(response: str) -> List[SRTBlock]:
    """Parse the response from Gemini API back into SRTBlock objects."""
    # Extract the corrected subtitles section
    match = re.search(r'【修正した字幕】\s*(.*)', response, re.DOTALL)
    if match:
        corrected_section = match.group(1).strip()
    else:
        # If the marker isn't found, use the entire response
        corrected_section = response.strip()
    
    # Split by double newline to get individual subtitle blocks
    raw_blocks = re.split(r'\n\n+', corrected_section)
    corrected_blocks = []
    
    for block in raw_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:  # Skip malformed blocks
            continue
        
        try:
            index = int(lines[0])
            timestamp = lines[1]
            text = '\n'.join(lines[2:])
            corrected_blocks.append(SRTBlock(index, timestamp, text))
        except ValueError:
            # Skip blocks with non-integer indices
            continue
    
    return corrected_blocks

def write_srt_file(srt_blocks: List[SRTBlock], file_path: str):
    """Write a list of SRTBlock objects to an SRT file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, block in enumerate(srt_blocks):
            # Update indices to ensure they're sequential
            block.index = i + 1
            f.write(f"{block}\n\n")

def correct_srt_with_gemini(input_file: str, output_file: str, batch_size: int = BATCH_SIZE):
    """
    Correct an SRT file using the Gemini AI Studio API.
    
    Args:
        input_file: Path to the input SRT file
        output_file: Path to save the corrected SRT file
        batch_size: Number of subtitle blocks to process in each API call
    """
    # Setup
    setup_gemini_api()
    
    # Parse the input SRT file
    print(f"Parsing SRT file: {input_file}")
    srt_blocks = parse_srt_file(input_file)
    print(f"Found {len(srt_blocks)} subtitle blocks")
    
    # Create batches
    batches = create_batches(srt_blocks, batch_size)
    print(f"Created {len(batches)} batches of size {batch_size}")
    
    # Process each batch
    corrected_blocks = []
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}...")
        
        # Generate prompt
        prompt = generate_correction_prompt(batch)
        
        try:
            # Call API
            response = call_gemini_api(prompt)
            
            # Parse response
            batch_corrected = parse_gemini_response(response)
            
            # Check if the batch was processed correctly
            if len(batch_corrected) < len(batch):
                print(f"Warning: Batch {i+1} returned fewer blocks than input ({len(batch_corrected)} vs {len(batch)})")
                # Use the original blocks where corrected ones are missing
                if len(batch_corrected) > 0:
                    # Map corrected blocks to original blocks by index
                    corrected_indices = [block.index for block in batch_corrected]
                    for original_block in batch:
                        if original_block.index not in corrected_indices:
                            batch_corrected.append(original_block)
                else:
                    # If no blocks were returned, use the original batch
                    batch_corrected = batch
            
            corrected_blocks.extend(batch_corrected)
            
            # Add a delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}")
            # Use the original blocks for this batch
            corrected_blocks.extend(batch)
    
    # Sort blocks by index
    corrected_blocks.sort(key=lambda block: block.index)
    
    # Write the corrected SRT file
    write_srt_file(corrected_blocks, output_file)
    print(f"Corrected SRT file saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Correct SRT files using Gemini AI Studio API")
    parser.add_argument("input_file", help="Path to the input SRT file")
    parser.add_argument("--output_file", "-o", help="Path to save the corrected SRT file (default: input_file_corrected.srt)")
    parser.add_argument("--batch_size", "-b", type=int, default=BATCH_SIZE, help=f"Number of subtitle blocks to process in each API call (default: {BATCH_SIZE})")
    parser.add_argument("--api_key", "-k", help="Gemini API key (overrides the one in the script)")
    
    args = parser.parse_args()
    
    # Set output file if not specified
    if not args.output_file:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_corrected{ext}"
    
    # Set API key if provided
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    # Check if API key is set
    if API_KEY == "YAIzaSyCztyJmLgpfj9ko5G5Y48AxnzP2Nj78xYM":
        print("Error: Gemini API key not set. Please provide an API key using the --api_key option.")
        return
    
    # Correct the SRT file
    correct_srt_with_gemini(args.input_file, args.output_file, args.batch_size)

if __name__ == "__main__":
    main()