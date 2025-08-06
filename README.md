# Assignment 02: Evaluating LLM Performance (Part 1)

## Setup
1. Clone/download this project.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
4. Run the script:
   ```bash
   python evaluate_llm_assignment.py
   ```

## Description
- Uses the provided product CSV file as data.
- Chapter 1: Evaluates with OpenAI (GPT-3.5-turbo).
- Chapter 2: Evaluates with Gemini (Gemini Pro).
- Prints example outputs for the first product row. 