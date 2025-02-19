# Calculus AI Chatbot

This project is a Calculus AI chatbot built using Python and LangChain. The chatbot is designed to assist users with calculus-related queries, providing explanations, solutions, and guidance.

## Project Setup

### Prerequisites

Make sure you have the following installed:

- Python (>=3.8)

### Setting up the environment

1. **Clone the repository**

   ```sh
   git clone https://github.com/Bulat-Mukhametdinov/cal-ai
   cd cal-ai
   ```

2. **Create a virtual environment**

   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment**

   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

4. **Open the `.env` file and add your API keys**

    ```env
    GROQ_API_KEY=your_openai_api_key_here
    ```

5. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```


## Helpful Links
- [groq available models](https://console.groq.com/docs/rate-limits)
- [langchain models usage]("https://python.langchain.com/docs/integrations/chat/")