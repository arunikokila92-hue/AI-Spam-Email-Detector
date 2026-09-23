import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

msg = input("Enter Email Text: ")

result = model.predict([msg])

print("Prediction:", result[0])