import os

print("Welcome to the COVID-19 Classifier Project!")
print("1. Train the model")
print("2. Run inference on an image")
print("3. Start FastAPI server")
choice = input("Enter your choice (1/2/3): ")

if choice == "1":
    os.system("python scripts/train.py")
elif choice == "2":
    image_path = input("Enter image path: ")
    os.system(f'python scripts/inference.py "{image_path}"')
elif choice == "3":
    os.system("uvicorn app:app --reload")
else:
    print("Invalid choice! Please enter 1, 2, or 3.")
