from groq import Groq

GROQ_API_KEY = "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"

client = Groq(api_key=GROQ_API_KEY)

models = client.models.list()

for model in models.data:
    print(model.id)