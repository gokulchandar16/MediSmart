from app.ai_engine import MediSmartAI


ai = MediSmartAI()

question = "What should I do if I have a fever?"

result = ai.answer(question)

print("\nQUESTION:")
print(result["question"])

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")
for source in result["sources"]:
    print("-", source["topic"])
