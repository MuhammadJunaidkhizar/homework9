import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes
from telegram.ext.filters import TEXT
from transformers import AutoModelForCausalLM, AutoTokenizer
import nest_asyncio

# Apply nest_asyncio for Colab or Jupyter environments
nest_asyncio.apply()

# Replace with your bot token
BOT_TOKEN = "8131659150:AAEOi-Fjzvu2qW1MRixAHlnPNwbiy957S5g"

# Load LLM
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("Loading model and tokenizer. This might take a while...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Model and tokenizer loaded successfully!")

# Function to process user input with LLM
def process_with_llm(user_message):
    try:
        # Prepare the input for the model
        formatted_input = f"User: {user_message}\nAssistant:"
        inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Generate a response with optimized parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Adjust for longer responses
            temperature=0.7,     # Control randomness
            top_p=0.9,           # Nucleus sampling
            num_return_sequences=1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "Assistant:" in response:
            response_text = response.split("Assistant:")[-1].strip()
        else:
            response_text = response.strip()

        return response_text
    except Exception as e:
        print(f"Error in LLM processing: {e}")
        return "I'm sorry, I couldn't process your request."

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the bot is started."""
    await update.message.reply_text("Hi! I'm your AI assistant. Ask me anything!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages and respond using the LLM."""
    user_message = update.message.text  # Get the user's input message

    # Log the received message
    print(f"Received message: {user_message}")

    # Avoid processing the bot's own messages
    if update.message.from_user.is_bot:
        print("Ignored a message from the bot itself.")
        return

    # Process the message with the LLM
    response = process_with_llm(user_message)

    # Send the response back to the user
    await update.message.reply_text(response)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors caused by updates."""
    print("Exception while handling an update:", exc_info=context.error)

# Main function to start the bot
def main():
    # Create the application
    application = Application.builder().token(BOT_TOKEN).build()

    # Command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(TEXT, handle_message))

    # Error handler
    application.add_error_handler(error_handler)

    # Run the bot
    print("Bot is starting...")
    application.run_polling()

if __name__ == "__main__":
    main()
