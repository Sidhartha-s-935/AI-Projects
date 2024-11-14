from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device="cuda:0")

sentences = ["I was not having a good day but well, it all worked out fine at the end!"]

model_outputs = classifier(sentences)
print(model_outputs[0])

