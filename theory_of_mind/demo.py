# demo.py
import sys
from apollo_email_assistant.agent import run_assistant_with_message, state

def main():
    print("Project Aurora Email Assistant Demo")
    print("==================================")
    print("Type 'exit' to quit the demo.")
    print()
    
    while True:
        user_input = input("Your message: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Thank you for using the Project Aurora Email Assistant.")
            break
        
        print("\nProcessing your request...\n")
        response = run_assistant_with_message(user_input, state)
        print(f"Assistant: {response}\n")

if __name__ == "__main__":
    main()