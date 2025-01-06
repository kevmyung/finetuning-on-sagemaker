import boto3
import json
import PyPDF2
import os
from typing import List, Dict

class PDFQAGenerator:
    def __init__(self):
        # Initialize Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-west-2'
        )
        
        # Model ID for Claude 3.5 Haiku
        self.model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def generate_qa_pairs(self, content: str, num_pairs: int = 5) -> List[Dict[str, str]]:
        """Generate Q&A pairs using Claude 3.5 model with Converse API"""
        system_prompt = f"""Generate exactly {num_pairs} Q&A pairs from the content below.

        STRICT FORMAT REQUIREMENTS:
        Return ONLY a JSON array like this exact format:
        [
            {{
                "question": "First question here?",
                "answer": "First answer here"
            }},
            {{
                "question": "Second question here?",
                "answer": "Second answer here"
            }}
        ]

        No other text or explanations - just the JSON array.
        """

        messages = [
            {
                "role": "user",
                "content": [{"text": f"Here's the content to generate Q&A pairs from: {content}"}]
            }
        ]

        try:
            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=messages
            )

            # Extract the response text from the message
            response_text = response['output']['message']['content'][0]['text']
            print(response_text)

            # Parse the JSON response
            qa_pairs = json.loads(response_text)
            return qa_pairs

        except Exception as e:
            raise Exception(f"Error generating Q&A pairs: {str(e)}")

    def chunk_text(self, text: str, chunk_size: int = 24000) -> List[str]:
        """Split text into chunks to handle large documents"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_large_document(self, content: str, output_file: str, num_pairs: int = 5) -> List[Dict[str, str]]:
        """Handle large documents by processing them in chunks and save to JSONL"""
        chunks = self.chunk_text(content, chunk_size=3000)
        all_qa_pairs = []
        pairs_per_chunk = max(1, num_pairs // len(chunks))

        # Open the JSONL file in append mode
        with open(output_file, 'a', encoding='utf-8') as jsonl_file:
            for i, chunk in enumerate(chunks):
                try:
                    chunk_pairs = self.generate_qa_pairs(chunk, pairs_per_chunk)
                    all_qa_pairs.extend(chunk_pairs)
                    
                    # Write each Q&A pair to the JSONL file immediately
                    for qa_pair in chunk_pairs:
                        # Add metadata about which chunk this came from
                        qa_pair['chunk_index'] = i
                        jsonl_file.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                        jsonl_file.flush()  # Ensure the data is written immediately
                    
                except Exception as e:
                    print(f"Error processing chunk {i}: {str(e)}")
                    continue

        return all_qa_pairs[:num_pairs]

def main():
    # Initialize the QA Generator
    qa_generator = PDFQAGenerator()

    # PDF file path and output file path
    pdf_path = "nova-manual.pdf"
    output_jsonl = "qa_pairs.jsonl"

    try:
        # Extract text from PDF
        print("Extracting text from PDF...")
        content = qa_generator.extract_text_from_pdf(pdf_path)

        # Generate Q&A pairs
        print("Generating Q&A pairs...")
        qa_pairs = qa_generator.process_large_document(content, output_jsonl, num_pairs=5)

        if not qa_pairs:
            print("No Q&A pairs were generated successfully.")
            return

        # Print the results
        print("\nGenerated Q&A Pairs:")
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\nPair {i}:")
            print(f"Q: {qa['question']}")
            print(f"A: {qa['answer']}")

        print(f"\nQ&A pairs have been saved to {output_jsonl}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
