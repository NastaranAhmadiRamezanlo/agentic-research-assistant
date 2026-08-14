from generator import generate_answer


question = "How much does University A cost?"

context = "The tuition fee is 12,000 euros per year."


answer = generate_answer(question, context)


print("Question:")
print(question)

print("\nAnswer:")
print(answer)