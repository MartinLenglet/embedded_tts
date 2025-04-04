#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:29:50 2022

@author: lengletm
"""

import os
import json
import yaml
import platform
import tkinter as tk
import tkinter.font as font

# Get the current OS
current_os = platform.system() 
# Optionally, you can print the OS
if current_os == "Windows":
    import simpleaudio as sa
else:
    from pydub import AudioSegment
    from pydub.playback import play

import loading_modules
import tts_utils
import audio_utils
import keyboards

# # Global variables to store the canvas and the circle figure
canvas_circle = None
canvas_circle_figure = None

def create_keyboard(key_board_options, entry, main_window=None, index_gst_token=0):
    global lbl_text_keyboard
    global entry_text_keyboard

    # Precise font_size
    myFont = "Helvetica {} bold".format(key_board_options["font_size"])

    # If no parent is provided, create a new window for the keyboard
    if main_window is None:
        window_keyboard = tk.Tk()
        window_keyboard.title(key_board_options["name_window"])
        window_keyboard.geometry("{}x{}".format(key_board_options["width"], key_board_options["height"]))
    else:
        # If a parent is provided, use the parent window or frame to embed the keyboard
        window_keyboard = tk.Frame(master=main_window)
        window_keyboard.grid(row=17+index_gst_token, column=0, columnspan=3, sticky=tk.NSEW)

    # Check if the entry text box should be shown
    if key_board_options.get("show_entry", True):
        entry_text_keyboard = tk.Entry(master=window_keyboard, width=44, state='readonly')
    else:
        entry_text_keyboard = None  # Set to None if hidden
    
    max_width_keyboard = 0

    for i_line, line in enumerate(keyboards.keys["Emmanuelle"]):
        tk.Grid.rowconfigure(window_keyboard,i_line+1,weight=1)
        for i_key, key in enumerate(line):
            max_width_keyboard = max(max_width_keyboard, i_key)
            tk.Grid.columnconfigure(window_keyboard,i_key,weight=1)
            key_label = key[0]
            if len(key) < 3:
                # Default Case: keys adds inputs to entry
                key_phon = key[1]
                current_button = tk.Button(
                    master=window_keyboard, 
                    text=key_label,
                    font=myFont,
                    width=key_board_options["max_button_width"],  # Set the max width of each button
                    wraplength=key_board_options["max_button_width"] * 10,  # Limit the text wrapping within the button
                    command= lambda current_phon=key_phon, current_label=key_label: [
                        entry.insert("end", "{} ".format(current_phon)), 
                        entry_readonly_insert(entry_text_keyboard, current_label, key_board_options) if entry_text_keyboard else play_prerecorded_phone(current_label, key_board_options)
                    ]
                )
            else:
                # Special Case: keys plays functions with specific entries (entries need to be in the global scope)
                key_function = getattr(keyboards, key[1])
                key_args = key[2]
                args = []
                for key_arg in key_args:
                    if isinstance(key_arg, int):
                        args.append(key_arg)
                    else:
                        args.append(globals()[key_arg])
                current_button = tk.Button(
                    master=window_keyboard, 
                    text=key_label,
                    font=myFont,
                    width=key_board_options["max_button_width"],  # Set the max width of each button
                    wraplength=key_board_options["max_button_width"] * 10,  # Limit the text wrapping within the button
                    command= lambda current_args=args, current_function=key_function: [current_function(current_args)]
                )
            current_button.grid(row=i_line+1, column=i_key, sticky=tk.NSEW, padx=key_board_options["key_margin_x"], pady=key_board_options["key_margin_y"])

    # Conditionally display the entry widget if the option is enabled
    if entry_text_keyboard:
        entry_text_keyboard['font'] = myFont
        entry_text_keyboard.grid(row=0, column=0, columnspan = max_width_keyboard+1, sticky=tk.W)
        entry_text_keyboard.grid_propagate(False)
    return window_keyboard

def create_gui(tts_config, device, default_tts, default_vocoder):
    global ent_text_input
    global lbl_audio_infos_audio_duration
    global lbl_audio_infos_tts_duration
    global lbl_audio_infos_vocoder_duration
    global lbl_audio_infos_denoiser_duration
    global lbl_audio_infos_synthesis_duration
    global gui_config
    global main_panel_config
    global TTS_CONFIG
    global window
    global lbl_gst_infos
    global canvas_circle
    global canvas_circle_figure
    
    TTS_CONFIG = tts_config
    gui_config = tts_config['GUI_config']
    main_panel_config = gui_config['main_panel']

    # Create the main window
    window = tk.Tk()
    window.title(main_panel_config['name_window'])
    window.geometry("{}x{}".format(main_panel_config["width"], main_panel_config["height"]))
    
    # Add specified TTS models
    max_buttons = max(len(tts_config["tts_models"]), len(tts_config["vocoder_models"]))
    lbl_TTS_model_selection = tk.Label(master=window, text="TTS :").grid(row=0, column=0, pady = 2)

    tts_index = 0
    list_tts_buttons = []
    for tts_model in tts_config["tts_models"]:
        tts_index += 1
        loading_script = getattr(loading_modules, tts_model["load_script"])
        gui_script = globals()[tts_model["gui_script"]]
        tts_loading_button = tk.Button(
            master=window, 
            text=tts_model["label"],
            command= lambda current_load_script=loading_script, current_tts=tts_model, id_button=tts_index, current_gui_script=gui_script: [current_load_script(current_tts, device), select_model_from_list(id_button, list_tts_buttons), tts_utils.update_selected_tts(id_button), current_gui_script(current_tts, main_panel_config)]
        )
        list_tts_buttons.append(tts_loading_button)
        tts_loading_button.grid(row=0, column=tts_index, columnspan=2, sticky=tk.NSEW)

        if tts_index==(default_tts+1):
            tts_loading_button.invoke()

    # Add specified Vocoder Models
    lbl_vocoder_model_selection = tk.Label(master=window, text="Vocodeur :").grid(row=1, column=0, pady = 2)
    vocoder_index = 0
    list_vocoder_buttons = []
    for vocoder_model in tts_config["vocoder_models"]:
        vocoder_index += 1
        loading_script = getattr(loading_modules, vocoder_model["load_script"])
        vocoder_loading_button = tk.Button(
            master=window, 
            text=vocoder_model["label"],
            command= lambda current_load_script=loading_script, current_vocoder=vocoder_model, id_button=vocoder_index: [current_load_script(current_vocoder, device), select_model_from_list(id_button, list_vocoder_buttons), tts_utils.update_selected_vocoder(id_button)]
        )
        list_vocoder_buttons.append(vocoder_loading_button)
        vocoder_loading_button.grid(row=1, column=vocoder_index, columnspan=2, sticky=tk.NSEW)
        
        if vocoder_index==(default_vocoder+1):
            vocoder_loading_button.invoke()

    # Add input field
    ent_text_input = tk.Entry(master=window, width=main_panel_config["input_width"])

    btn_syn_audio = tk.Button(
        master=window, 
        text="Synthèse",
        command= lambda is_gui=True, tts_global_conf=tts_config: audio_utils.syn_audio(is_gui, tts_global_conf, gui_control=get_gui_controls())
    )
    
    if not gui_config["detach_keyboard"] and gui_config["keyboard_options"]["show_entry"]:
        lbl_text_input = tk.Label(master=window, text="Input Text").grid(row=7, column=0, pady = 4)
    
        ent_text_input.grid(row=7, column=1)
        ent_text_input.bind("<Return>", lambda is_gui=True, tts_global_conf=tts_config: audio_utils.syn_audio(is_gui, tts_global_conf, gui_control=get_gui_controls()))

        btn_syn_audio.grid(row=7, column=2)
    
    # Add audio infos
    if main_panel_config["add_audio_infos"]:

        lbl_audio_infos_audio_duration = tk.Label(master=window, text="Durée audio : 0.000s")
        lbl_audio_infos_audio_duration.grid(row=8, column=0, columnspan=max_buttons+2)

        # Add a Canvas next to the lbl_audio_infos_audio_duration to draw a circle
        canvas_circle = tk.Canvas(master=window, width=20, height=20)
        canvas_circle.grid(row=8, column=2)  # Positioned next to the label
        # Create a circle on the canvas
        canvas_circle_figure = canvas_circle.create_oval(2, 2, 18, 18, fill="gray")  # Initial color set to gray

        lbl_audio_infos_tts_duration = tk.Label(master=window, text="Durée TTS : 0.000s | 0% de la durée audio")
        lbl_audio_infos_tts_duration.grid(row=9, column=0, columnspan=max_buttons+2)
        lbl_audio_infos_vocoder_duration = tk.Label(master=window, text="Durée Vocodeur : 0.000s | 0% de la durée audio")
        lbl_audio_infos_vocoder_duration.grid(row=10, column=0, columnspan=max_buttons+2)
        lbl_audio_infos_denoiser_duration = tk.Label(master=window, text="Durée Denoiser : 0.000s | 0% de la durée audio")
        lbl_audio_infos_denoiser_duration.grid(row=11, column=0, columnspan=max_buttons+2)
        lbl_audio_infos_synthesis_duration = tk.Label(master=window, text="Durée Totale Synthèse : 0.000s | 0% de la durée audio")
        lbl_audio_infos_synthesis_duration.grid(row=12, column=0, columnspan=max_buttons+2)

    # Add audio infos
    if main_panel_config["add_GST_infos"]:
        lbl_gst_infos = {}
        label_gst_title = tk.Label(master=window, text="\nGST weights\n")
        label_gst_title.grid(row=14, column=0, columnspan=max_buttons+2)
        for index_gst_token, gst_token in enumerate([*tts_config['tts_models'][0]['gst_token_list']]):
            lbl_gst_infos[gst_token] = tk.Label(master=window, text="{}: 0.00".format(gst_token))
            lbl_gst_infos[gst_token].grid(row=15+index_gst_token, column=0, columnspan=max_buttons+2)
    else:
        index_gst_token = 0

    # Add replay button
    if main_panel_config["add_play_button"]:
        btn_replay_audio = tk.Button(
            master=window, 
            text="Play",
            command=audio_utils.play_audio
        ).grid(row=16+index_gst_token, column=0, columnspan=max_buttons+2)

    if gui_config["add_keyboard"]:
        if gui_config["detach_keyboard"]:
            window_keyboard = create_keyboard(gui_config["keyboard_options"], ent_text_input)
            window_keyboard.mainloop()
        else:
            window_keyboard = create_keyboard(gui_config["keyboard_options"], ent_text_input, window, index_gst_token)
    
    window.mainloop()

def update_audio_infos(audio_duration, tts_inference_duration, vocoder_inference_duration, denoiser_inference_duration):
    if main_panel_config["add_audio_infos"]:
        total_inference_duration = tts_inference_duration + vocoder_inference_duration + denoiser_inference_duration
        lbl_audio_infos_audio_duration["text"] = "Durée audio : {:.3f}s".format(audio_duration)
        lbl_audio_infos_tts_duration["text"] = "Durée TTS : {:.3f}s | {:.0f}% de la durée audio".format(tts_inference_duration, 100*tts_inference_duration/audio_duration)
        lbl_audio_infos_vocoder_duration["text"] = "Durée Vocodeur : {:.3f}s | {:.0f}% de la durée audio".format(vocoder_inference_duration, 100*vocoder_inference_duration/audio_duration)
        lbl_audio_infos_denoiser_duration["text"] = "Durée Denoiser : {:.3f}s | {:.0f}% de la durée audio".format(denoiser_inference_duration, 100*denoiser_inference_duration/audio_duration)
        lbl_audio_infos_synthesis_duration["text"] = "Durée Totale Synthèse : {:.3f}s | {:.0f}% de la durée audio".format(total_inference_duration, 100*total_inference_duration/audio_duration)

def update_GST_infos(GST_weights):
    if main_panel_config["add_GST_infos"]:
        for lbl_gst_info, token_weight in zip(lbl_gst_infos.items(), GST_weights):
            (token_name, label_gst) = lbl_gst_info
            label_gst["text"] = "{}: {:.2f}".format(token_name, token_weight[0])

def label_insert(label, insert):
    current_text = label.cget("text")
    label["text"] = "{} {} ".format(current_text, insert)

def entry_readonly_insert(entry, insert, key_board_options):
    entry['state'] = 'normal'
    entry.insert("end", "{} ".format(insert))
    entry.xview_moveto(1)
    entry['state'] = 'readonly'

    # Play sound
    play_prerecorded_phone(insert, key_board_options)

def play_prerecorded_phone(phone, keyboard_config):
    if keyboard_config["play_phone"]:
        # Play the preloaded audio file
        audio_file_path = os.path.join("audio_keyboards", keyboard_config["keys"], f"{phone}.wav")
        # Get the current OS
        current_os = platform.system() 
        # Optionally, you can print the OS
        if current_os == "Windows":
            wave_obj = sa.WaveObject.from_wave_file(audio_file_path)
            play_obj = wave_obj.play()
        else:
            audio = AudioSegment.from_wav(audio_file_path)
            play(audio)
    
def select_model_from_list(id_button, list_buttons):
    # Reset background of all buttons
    index_button = 0
    for button in list_buttons:
        index_button += 1
        if index_button == id_button:
            button["bg"] = "yellow"
        else:
            button["bg"] = "#f0f0f0"

def get_gui_controls():
    speaker_id = speaker_selection.get()
    pitch_control = pitch_slider.get()
    energy_control = energy_slider.get()
    speed_control = speed_slider.get()
    pitch_control_bias = pitch_bias_slider.get()
    energy_control_bias = energy_bias_slider.get()
    speed_control_bias = speed_bias_slider.get()
    pause_control_bias = pause_bias_slider.get()
    liaison_control_bias = liaison_bias_slider.get()
    gst_token_index = gst_token_selection.get()
    style_intensity_control = style_intensity_slider.get()
    styleTag_control = ent_styleTag_input.get()

    result = [
        speaker_id, 
        pitch_control, 
        energy_control, 
        speed_control, 
        pitch_control_bias, 
        energy_control_bias, 
        speed_control_bias,
        pause_control_bias,
        liaison_control_bias,
        gst_token_index,
        style_intensity_control,
        styleTag_control,
    ]

    return result

# Function to update the color of the circle
def get_canvas_circle():
    return canvas_circle, canvas_circle_figure

def update_circle_color(color, canvas_circle, canvas_circle_figure):
    print(f"Changing circle color to {color}")
    canvas_circle.itemconfig(canvas_circle_figure, fill=color)

def gui_fastspeech2(tts_config, main_panel_config):
    global speaker_selection
    global pitch_slider
    global energy_slider
    global speed_slider
    global pitch_bias_slider
    global energy_bias_slider
    global speed_bias_slider
    global pause_bias_slider
    global liaison_bias_slider
    global gst_token_selection
    global style_intensity_slider
    global ent_styleTag_input
    global canvas

    sub_row_index = 0
    default_args = tts_config['default_args']
    preprocess_config = yaml.load(
        open(os.path.join(tts_config['folder'], default_args["preprocess_config"]), "r"), Loader=yaml.FullLoader
    )
    speakers_location = os.path.join(preprocess_config['path']['preprocessed_path'], "speakers.json")

    # Create Options Frame with Scrollbar
    frame = tk.Frame(window, highlightbackground="black", highlightthickness=2)
    frame.grid(row=2, column=0, columnspan=3, sticky='nw')
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_propagate(False)
    canvas = tk.Canvas(frame)
    canvas.grid(row=0, column=0, sticky='news')
    vsb = tk.Scrollbar(frame, orient='vertical', command=canvas.yview)
    vsb.grid(row=0,column=1, sticky='ns')
    canvas.configure(yscrollcommand=vsb.set)
    frame_options = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame_options, anchor='nw')
    
    # ~ lbl_tts_options_selection = tk.Label(master=frame_options, text="TTS options :", font='Helvetica 15 underline').grid(row=0, column=0, rowspan = 4)
    
    # Select default values
    speaker_selection = tk.IntVar(frame)
    speaker_selection.set(default_args['speaker_id'])

    with open(speakers_location, "r") as f:
        speaker_list = json.load(f)

    # Speaker radio buttons
    lbl_speaker_selection = tk.Label(master=frame_options, text="Speaker :").grid(row=sub_row_index, column=0)
    index_speaker = 0
    for speaker in speaker_list:
        tk_radio_button_speaker = tk.Radiobutton(
            master=frame_options, 
            text=speaker, 
            variable=speaker_selection, 
            value=index_speaker, 
            command=None, 
        )
        tk_radio_button_speaker.grid(row=sub_row_index, column=1+index_speaker)
        index_speaker += 1
    sub_row_index += 1
    
    # Select default values
    gst_token_selection = tk.IntVar(frame)
    gst_token_selection.set(default_args['gst_token_index'])

    # Free StyleTag input field
    ent_styleTag_input = tk.Entry(master=frame_options, width=main_panel_config["input_width"])
    if tts_config['gui_styleTag_control']:
        lbl_styleTag_input = tk.Label(master=frame_options, text="StyleTag :").grid(row=sub_row_index, column=0)
        ent_styleTag_input.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)
        sub_row_index += 1

    # GST token radio buttons
    if tts_config['gui_style_control']:
        lbl_gst_token_selection = tk.Label(master=frame_options, text="Style :").grid(row=sub_row_index, column=0)
    
        index_gst_token = 0
        for gst_token in [*tts_config['gst_token_list']]:
            # print(gst_token)
            tk_radio_button_gst_token = tk.Radiobutton(master=frame_options, text=gst_token, variable=gst_token_selection, value=index_gst_token, command=None)
            tk_radio_button_gst_token.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)
            index_gst_token += 1
            sub_row_index += 1

    # Style Intensity Slider
    style_intensity_slider = tk.Scale(frame_options, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.05)
    lbl_style_intensity = tk.Label(master=frame_options, text="Style Intensity:").grid(row=sub_row_index, column=0)
    style_intensity_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)
    style_intensity_slider.set(default_args['style_intensity'])
    sub_row_index += 1

    # Pitch Slider
    pitch_slider = tk.Scale(frame_options, from_=-15, to=15, orient=tk.HORIZONTAL, resolution=1)
    lbl_pitch_selection = tk.Label(master=frame_options, text="Pitch (semitones):").grid(row=sub_row_index, column=0)
    pitch_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)
    pitch_slider.set(default_args['pitch_control'])
    sub_row_index += 1

    # Energy Slider
    energy_slider = tk.Scale(frame_options, from_=-20, to=20, orient=tk.HORIZONTAL, resolution=1)
    lbl_energy_selection = tk.Label(master=frame_options, text="Energy (dB):").grid(row=sub_row_index, column=0)
    energy_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)
    energy_slider.set(default_args['energy_control'])
    sub_row_index += 1

    # Speed Slider
    speed_slider = tk.Scale(frame_options, from_=0.5, to=1.5, orient=tk.HORIZONTAL, resolution=0.1)
    lbl_speed_selection = tk.Label(master=frame_options, text="Speed (coef):").grid(row=sub_row_index, column=0)
    speed_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)
    speed_slider.set(default_args['duration_control'])
    sub_row_index += 1

    # Pitch Bias Slider
    pitch_bias_slider = tk.Scale(frame_options, from_=-6, to=6, orient=tk.HORIZONTAL, resolution=0.5)
    pitch_bias_slider.set(default_args['pitch_control_bias'])
    sub_row_index += 1

    # Energy Bias Slider
    energy_bias_slider = tk.Scale(frame_options, from_=-5, to=5, orient=tk.HORIZONTAL, resolution=1)
    energy_bias_slider.set(default_args['energy_control_bias'])
    sub_row_index += 1

    # Speed Bias Slider
    speed_bias_slider = tk.Scale(frame_options, from_=0.5, to=1.5, orient=tk.HORIZONTAL, resolution=0.1)
    speed_bias_slider.set(default_args['duration_control_bias'])
    sub_row_index += 1
    
    # Pause Bias Slider
    pause_bias_slider = tk.Scale(frame_options, from_=-2, to=2, orient=tk.HORIZONTAL, resolution=0.1)
    pause_bias_slider.set(default_args['pause_control_bias'])
    sub_row_index += 1

    # Liaison Bias Slider
    liaison_bias_slider = tk.Scale(frame_options, from_=-2, to=2, orient=tk.HORIZONTAL, resolution=0.1)
    liaison_bias_slider.set(default_args['liaison_control_bias'])
    sub_row_index += 1

    if tts_config['gui_control_bias']:
        lbl_speed_selection = tk.Label(master=frame_options, text="Pitch Bias (semitones):").grid(row=sub_row_index, column=0)
        pitch_bias_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)

        lbl_speed_selection = tk.Label(master=frame_options, text="Energy Bias (dB):").grid(row=sub_row_index, column=0)
        energy_bias_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)

        lbl_speed_selection = tk.Label(master=frame_options, text="Speed Bias (coef):").grid(row=sub_row_index, column=0)
        speed_bias_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)

        lbl_speed_selection = tk.Label(master=frame_options, text="Pause Bias:").grid(row=sub_row_index, column=0)
        pause_bias_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)
        
        lbl_speed_selection = tk.Label(master=frame_options, text="Liaison Bias:").grid(row=sub_row_index, column=0)
        liaison_bias_slider.grid(row=sub_row_index, column=1, columnspan=1+index_speaker)

    # Add scrollbar
    frame_options.update_idletasks()
    width_canvas = main_panel_config["control_width"]
    height_canvas = main_panel_config["control_height"]
    frame.config(width=width_canvas + vsb.winfo_width(), height=height_canvas)
    canvas.config(scrollregion=canvas.bbox("all"))
    # Make Scrollbar usable with mouse wheel
    canvas.bind('<Enter>', bound_to_mouse_wheel)
    canvas.bind('<Leave>', unbound_to_mouse_wheel)
    
def bound_to_mouse_wheel(event):
    canvas.bind_all('<Button-4>', mouse_wheel_up)
    canvas.bind_all('<Button-5>', mouse_wheel_down)
    
def unbound_to_mouse_wheel(event):
    canvas.unbind_all('<Button-4>')
    canvas.unbind_all('<Button-5>')
    
def mouse_wheel_up(event):
    canvas.yview_scroll(-1, 'units')
    
def mouse_wheel_down(event):
    canvas.yview_scroll(1, 'units')
