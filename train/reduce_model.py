import torch
import copy
from transformers import AutoConfig
from qwen_asr import Qwen3ASRModel

def get_model_size(model):
    return sum(p.numel() for p in model.parameters())

def create_smaller_architecture(original_model_path="Qwen/Qwen3-ASR-0.6B"):
    print(f"Loading original model configuration from {original_model_path}...")
    
    # On charge juste la config
    config = AutoConfig.from_pretrained(original_model_path, trust_remote_code=True)
    
    # Affichons la taille de base (sur l'architecture non instanciée, on compte juste approximativement
    # mais on va l'instancier pour avoir le vrai compte si besoin)
    
    # Analogie de ce qu'on peut réduire :
    # Qwen3-ASR-0.6B a typiquement pour l'audio un encodeur et pour le texte un decodeur LLM (dans thinker_config).
    print(f"Original LLM Layers: {config.thinker_config.text_config.num_hidden_layers}")
    # --- Stratégie de réduction ---
    small_config = copy.deepcopy(config)
    
    # 1. On fixe les couches selon ta proposition
    small_config.thinker_config.text_config.num_hidden_layers = 12
    small_config.thinker_config.audio_config.num_hidden_layers = 9

    # 2. Pour descendre vraiment en taille (car le vocabulaire 151k prend énormément de place),
    # on doit réduire considérablement les dimensions cachées.
    # Dimensions d'origine Probables : hidden = ~896 / 1536... on va diviser
    orig_text_hidden = config.thinker_config.text_config.hidden_size
    orig_text_interm = config.thinker_config.text_config.intermediate_size
    
    # On réduit le hidden_size (attention il faut que hidden_size soit un multiple de num_attention_heads)
    # Ex: si heads=14, on peut prendre hidden=336 ou 448
    heads = config.thinker_config.text_config.num_attention_heads
    new_hidden = heads * 16 # ex: 14 * 16 = 224 (très petit)
    
    small_config.thinker_config.text_config.hidden_size = new_hidden
    small_config.thinker_config.text_config.intermediate_size = new_hidden * 4
    
    # On réduit aussi l'audio
    # /!\ Il faut que d_model soit divisible par encoder_attention_heads (par défaut = 20)
    audio_heads = config.thinker_config.audio_config.encoder_attention_heads
    small_config.thinker_config.audio_config.d_model = audio_heads * 24 # ex: 20 * 24 = 480
    small_config.thinker_config.audio_config.encoder_ffn_dim = 2048
    
    # Prise en compte de tous les paramètres modifiés dans audio encoder config
    small_config.thinker_config.audio_config.encoder_layers = 9
    
    print("\n--- Reduced Model Architecture ---")
    print(f"New LLM Layers: {small_config.thinker_config.text_config.num_hidden_layers}")
    print(f"New LLM Hidden Size: {small_config.thinker_config.text_config.hidden_size} (was {orig_text_hidden})")
    print(f"New Audio Layers: {small_config.thinker_config.audio_config.num_hidden_layers}")
    print(f"New Audio d_model: {small_config.thinker_config.audio_config.d_model} (was {config.thinker_config.audio_config.d_model})")
    print(f"New Vocab Size: {small_config.thinker_config.text_config.vocab_size}")

    return small_config

def main():
    # 1. On va d'abord analyser Qwen/Qwen3-ASR-0.6B pour voir son architecture
    # On instantie un modèle complet original pour voir le nombre de paramètres
    print("Loading original model (meta) to count parameters...")
    from accelerate import init_empty_weights
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
    
    config = AutoConfig.from_pretrained("Qwen/Qwen3-ASR-0.6B", trust_remote_code=True)
    with init_empty_weights():
        model = Qwen3ASRForConditionalGeneration(config)
    
    orig_params = get_model_size(model)
    print(f"Original Model Parameters: {orig_params / 1e6:.2f} M (approx 0.6B)")
    
    # 2. On crée la mini config
    small_config = create_smaller_architecture("Qwen/Qwen3-ASR-0.6B")
    
    with init_empty_weights():
        small_model = Qwen3ASRForConditionalGeneration(small_config)
        
    small_params = get_model_size(small_model)
    print(f"Small Model Parameters: {small_params / 1e6:.2f} M (Target: ~0.1B)")
    

if __name__ == "__main__":
    main()
