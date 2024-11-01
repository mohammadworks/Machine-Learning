import telebot
import subprocess
import os

TOKEN = 'YOUR_BOT_TOKEN'
bot = telebot.TeleBot(TOKEN)

input_folder = "input_images/"
output_folder = "output_images/"

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    input_path = os.path.join(input_folder, f"{message.from_user.id}_input.jpg")
    output_path = os.path.join(f"{output_folder}final_output/", f"{message.from_user.id}_input.png")
    with open(input_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    command = [
        "python", "run.py",
        "--input_folder", input_folder,
        "--output_folder", output_folder,
        "--GPU", "0",
        "--with_scratch"
    ]
    try:
        subprocess.run(command, check=True)
        if os.path.exists(output_path):
            with open(output_path, 'rb') as processed_photo:
                bot.send_photo(message.chat.id, processed_photo)
        else:
            bot.send_message(message.chat.id, "Error: Output file not found.")
    except subprocess.CalledProcessError as e:
        bot.send_message(message.chat.id, "An error occurred while processing the image.")
        print("Error:", e)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_message(message.chat.id, "Please upload an image for processing.")


bot.polling()
