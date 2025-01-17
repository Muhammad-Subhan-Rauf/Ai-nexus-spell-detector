import os
import pygame

# Define the dictionary of spells
spells = {
    1: "Alohomora",
    2: "Avada Kedavra",
    3: "Expelliarmus",
    4: "Expecto Patronum",
    5: "Lumos",
    6: "Nox",
    7: "Wingardium Leviosa",
    8: "Stupefy",
    9: "Library Lockus",
    10: "Professorious",
    11: "Mosquito Expellio",
    12: "Plagiarismus Detectio",
    13: "Basuri Melodico",
    14: "Giftus Appearus",
    15: "Manaclus Bindio",
    16: "Ancestor Callium",
    17: "Dividus Zerox",
    18: "Attendanceus Finalus",
    19: "Deadlineius Erasum",
    20: "Stressius",
    21: "Tempus Forwarius",
    22: "Tempus Reversio",
    23: "Flyhighus Ascendo",
    24: "Flylowus Decendo",
    25: "Valentino"
}

# Path to the folder containing the sound files
sounds_folder = "sounds"

# Initialize Pygame mixer
pygame.mixer.init()

def play_spell_sound(index):
    if index in spells:
        spell_name = spells[index]
        print(f"Spell: {spell_name}")

        # Construct the sound file path
        sound_file = os.path.join(sounds_folder, f"{index}.mp3")
        
        # Check if the sound file exists
        if os.path.exists(sound_file):
            try:
                # Load and play the sound
                pygame.mixer.music.load(sound_file)
                pygame.mixer.music.play()

                # Wait for the sound to finish playing
                while pygame.mixer.music.get_busy():
                    pass
            except Exception as e:
                print(f"Error playing sound: {e}")
        else:
            print(f"Sound file for '{spell_name}' not found: {sound_file}")
    else:
        print("Invalid spell index. Please provide a number between 1 and 25.")

# Example usage
# while True:
#     try:
#         index = int(input("Enter the spell index (1-25): "))
#         play_spell_sound(index)
#     except ValueError:
#         print("Please enter a valid number.")
