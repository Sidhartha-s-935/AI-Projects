from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summary(text:str):
    inputs = tokenizer.encode("summarize: "+text, return_tensors="pt", max_length=1024,truncation=True)
    gen = model.generate(inputs, max_length=150,num_beams=4,early_stopping=True)
    return tokenizer.decode(gen[0],skip_special_tokens=True)

original = input("Enter your yapology: ")
duplicate = summary(original)
print("Conscise Yap: ", duplicate)
