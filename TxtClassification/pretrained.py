from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device="cuda:0")

x = input("Enter your sentence: ")
sentences = []
sentences.append(x)
model_outputs = classifier(sentences)
print(model_outputs[0])

