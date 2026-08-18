# Đếm token thật với thư viện tiktoken của OpenAI
# Cài: pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # tokenizer dòng GPT
text_vie = "Xin chào, mình học câu lệnh cho AI 2026!"

tokens = enc.encode(text_vie)
print("Số token:", len(tokens))          # ví dụ: ~13 token
print("Token IDs:", tokens)              # [mảng số nguyên]
print("Giải mã lại:", enc.decode(tokens))  # ra đúng câu gốc

text_eng = "Hi, i'm learning Prompt AI 2026!"
tokens = enc.encode(text_eng)
print("Số token:", len(tokens))         
print("Token IDs:", tokens)              
print("Giải mã lại:", enc.decode(tokens))  