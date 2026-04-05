#Explorations; ne fait pas partie du projet principal;  a ignorer.

model_path = "C:/path/to/Documents/models/LFM2.5-1.2B-Thinking-Q8_0.gguf"

from llama_cpp import Llama
llm = Llama(
      model_path="path/to/llama-2/llama-model.gguf",
      chat_format="llama-2"
)
llm.create_chat_completion(
      messages = [
          {"role": "system", "content": "You are a clown."},
          {
              "role": "user",
              "content": "What's up."
          }
      ]
)