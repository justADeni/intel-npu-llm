import time
import re
import os
import copy
import warnings
import subprocess
from colorama import Fore
import openvino
import openvino_genai
from huggingface_hub import model_info
from huggingface_hub.utils import HfHubHTTPError, HFValidationError

# Helper functions

def check_for_NPU():
    if "NPU" not in openvino.Core().available_devices:
        print(Fore.RED + "NPU was not detected. Have you installed the latest NPU driver?", flush=True)
        exit()

def streamer(subword):
    print(subword, end='', flush=True)
    return False

def select_from_list(options = [], question=Fore.GREEN + "Select a number: " + Fore.RESET, allow_custom=False):
    if not options:
        raise ValueError(Fore.RED + "The list is empty.")

    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}", flush=True)

    while True:
        try:
            choice = input(question)
            if choice == "exit":
                exit()
            
            choice = int(choice)
            if 1 <= choice <= len(options):
                return strip_ansi_colors(options[choice - 1])
            else:
                print(Fore.RED + "Invalid number. Please try again.", flush=True)
        except ValueError:
            if allow_custom:
                if check_model_exists(choice):
                    return choice
            else:    
                print(Fore.RED + "Please enter a valid number.", flush=True)

def check_model_exists(model_id):
    try:
        # Attempt to fetch model information
        info = model_info(model_id)
        return True
    except (HfHubHTTPError, HFValidationError) as e:
        if "404" in str(e):
            print(Fore.RED + f"Model '{model_id}' not found. Are you connected to the internet?" + Fore.RESET, flush=True)
            return False
        elif "namespace/repo_name" in str(e):
            print(Fore.RED + f"Repo id must be in the form 'repo_name' or 'namespace/repo_name'" + Fore.RESET, flush=True)
            return False
        else:
            print(f"An error occurred: {e}")
            raise

# Directory containing the models (relative to script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")

def check_gated(model_name):
    try:
        info = model_info(model_name)
        return (info.cardData and info.cardData.get("gated"))
    except Exception as e:
        print(Fore.RED + "Error checking model. Are you connected to the internet?" + Fore.RESET)
        exit()

def register_token():
    repeated = False
    while get_command_output("huggingface-cli whoami") == "Not logged in":
        if repeated:
            print(Fore.RED + "Error logging in." + Fore.RESET, flush=True)
        else:
            repeated = True
            
        token = input(Fore.GREEN + "Enter huggingface token: " + Fore.RESET)
        if token == "exit":
            exit()
        os.system("huggingface-cli login --token " + token)    

def get_command_output(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout.strip()

def local_models():
    models = []
    try:
        for vendor in os.listdir(models_dir):
            for model in os.listdir(os.path.join(models_dir, vendor)):
                models.append(f"{vendor}/{model}")
        if not models:
            return []
        return models
    except FileNotFoundError:
        return []        

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def strip_ansi_colors(string):
    return ansi_escape.sub('', string)

def streamer(subword):
    print(subword, end='', flush=True)
    return False

def get_valid_number(prompt, default_value, min_value, max_value):
    while True:
        try:
            user_input = input(prompt + "\n")
            if user_input == "":
                return default_value
            if user_input == "exit":
                exit()
            user_input = int(user_input)
            if min_value <= user_input <= max_value:
                return user_input
            else:
                print(Fore.RED + f"Please enter a number between {min_value} and {max_value}." + Fore.RESET, flush=True)
        except ValueError:
            print(Fore.RED + "Invalid input. Please enter a valid number." + Fore.RESET, flush=True)

# From https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/genai-guide-npu.html
model_selection = [
"meta-llama/Meta-Llama-3.1-8B-Instruct",
"microsoft/Phi-3-mini-4k-instruct",
"Qwen/Qwen2-7B",
"mistralai/Mistral-7B-Instruct-v0.2",
"openbmb/MiniCPM-1B-sft-bf16",
"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
# For some reason, gptq models just won't run, and auto-gptq won't install
#"TheBloke/Llama-2-7B-Chat-GPTQ",
#"Qwen/Qwen2-7B-Instruct-GPTQ-Int4" 
]

# The main part of the script below

def download_and_or_select_model():
    print(Fore.GREEN + "Select a model." + Fore.RESET, flush=True)

    l = local_models()
    c = copy.deepcopy(model_selection)
    available_to_download = [item for item in c if not any(sub in item for sub in l)]

    c += [s for s in l if not any(s in u for u in c)]
    c = [ Fore.CYAN + s + Fore.RESET if s not in available_to_download else Fore.RESET + s for s in c]

    selected = select_from_list(c, Fore.GREEN + "Select a number or paste custom model name: " + Fore.RESET, allow_custom=True)
        
    if selected not in l:
        if check_gated(selected):
            register_token()

        loaded = False
        while not loaded:
            try:
                print(Fore.GREEN + "Downloading and quantizing selected model to INT4." + Fore.RESET, flush=True)
                print(os.system("optimum-cli export openvino -m " + selected + " --weight-format int4 --sym --ratio 1.0 --group-size 128 --trust-remote-code --task text-generation-with-past --cache_dir " + os.path.join(script_dir,"download_cache") + " models/" + selected))
                loaded = True    
            except ValueError as e:
                print(Fore.RED + f"Error while quantizing {selected}:" + Fore.RESET + f"{e}", flush=True)

    return selected

def load(model_name, model_path, prompt_length):
    is_cached = os.path.isdir(os.path.join(script_dir, "npu_cache", model_name))
    if is_cached:
        loading_text = "Loading the NPU compiled model from cache."
        loaded_text = "Model loaded in "
    if not is_cached:
        print(Fore.MAGENTA + "Since you're running it for the first time, the model will be compiled and cached, which may take a while (up to tens of minutes), especially for larger models. Subsequent starts will be much faster (tens of seconds), though still resource intensive.", flush=True)
        loading_text = "Compiling and caching the model for NPU."
        loaded_text = "Model compiled in "

    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    print(Fore.GREEN + loading_text + Fore.RESET, flush=True)
    model_load_start = time.time()
    pipeline_config = { 
        "NPUW_CACHE_DIR": os.path.join(script_dir, "npu_cache", model_name.split("/")[0], model_name.split("/")[1]),
        "GENERATE_HINT": "BEST_PERF",
        "MAX_PROMPT_LEN": prompt_length }
    pipe = openvino_genai.LLMPipeline(os.path.join(script_dir, model_path), 'NPU', pipeline_config)
    model_load_stop = time.time()
    print(Fore.GREEN + loaded_text + str(round(model_load_stop - model_load_start,1)) + " seconds. \n" + Fore.RESET, flush=True)
    return pipe

def generate(pipe, config):
    while True:
        try:
            pipe.start_chat()
            while True:
                try:
                    prompt = input("prompt: ")
                except EOFError:
                    break
                
                if prompt == "exit":
                    exit()
                if prompt == "reset":
                    # Reminds me of "I raised that boy" meme
                    raise RuntimeError("manual")

                gen_start = time.time()
                response = pipe.generate(prompt, config, streamer)
                gen_stop = time.time()
                # I know this isn't a super accurate measure, but openvino.PerfMetrics refused to work
                print("\n" + str(round((len(response)/(gen_stop - gen_start))/4,1)) + "t/s")
                print('\n----------')
            pipe.finish_chat()
        except RuntimeError as e:
            if str(e) == "manual":
                print(Fore.GREEN + "Chat reset." + Fore.RESET, flush=True)
            else:
                print(Fore.MAGENTA + "Chat got reset due to overflowing max prompt length." + Fore.RESET, flush=True)
            pipe.finish_chat()

def main():
    check_for_NPU()
    model_name = download_and_or_select_model()
    model_path = os.path.join("models", model_name)
    prompt_length = get_valid_number(Fore.GREEN + "\nPick context length between 256 and 16384. This number includes both your prompts and llm responses, until it overflows and chat resets. Larger numbers will take longer to compile, run and will consume more memory. (1024)" + Fore.RESET, 1024, 256, 16384)
    pipe = load(model_name, model_path, prompt_length)

    config = openvino_genai.GenerationConfig()
    config.do_sample = False
    config.top_k = 50
    config.top_p = 0.9
    config.repetition_penalty = 1.2
    config.no_repeat_ngram_size = 3
    config.temperature = 0.75

    print(Fore.GREEN + "Chat commands: \nexit - unload the model and exit the script \nreset - resets the chat context manually\n" + Fore.RESET, flush=True)
    generate(pipe, config)

if '__main__' == __name__:
    main()