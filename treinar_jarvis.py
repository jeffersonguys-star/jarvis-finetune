# treinar_jarvis.py — fine-tuning QLoRA do Qwen3-8B no dataset_jarvis.jsonl
#
# Instala as dependências ANTES, numa célula separada do Colab:
#   !pip install -q -U transformers peft trl bitsandbytes accelerate datasets
#
# Isso NÃO foi testado rodando de verdade (sem GPU/internet aqui) — rode e
# me manda o erro exato se travar em algum lugar, corrijo na hora.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduz fragmentação de VRAM

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODELO_BASE = "Qwen/Qwen3-4B"  # trocado de 8B pra 4B — 8B não sobrava memória suficiente na T4 (16GB) pra sequência de 6-7 mil tokens
ARQUIVO_DATASET = "dataset_jarvis.jsonl"
PASTA_SAIDA = "jarvis-qwen3-lora"
TAMANHO_MAX_SEQUENCIA = 5120  # contexto de ferramentas foi limitado no gerador (máx 42, era até 68) — deve cobrir tudo com folga agora

# ---------------------------------------------------------------------------
# 1) Tokenizer + modelo base em 4-bit (é isso que faz o 8B caber numa GPU
#    pequena tipo a T4 do Colab grátis, que tem 16GB)
# ---------------------------------------------------------------------------
print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Carregando modelo base (baixa ~5GB na primeira vez, demora alguns minutos)...")
model = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ---------------------------------------------------------------------------
# 2) LoRA — só treina uma fração pequena dos pesos, não o modelo inteiro
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# 3) Dataset — usa o chat template do PRÓPRIO Qwen3 (ele já sabe formatar
#    'tools' e 'tool_calls' no formato certo, não precisamos escrever isso)
# ---------------------------------------------------------------------------
print("Carregando dataset...")
dataset = load_dataset("json", data_files=ARQUIVO_DATASET, split="train")

def formatar(exemplo):
    texto = tokenizer.apply_chat_template(
        exemplo["messages"],
        tools=exemplo["tools"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": texto}

dataset = dataset.map(formatar, remove_columns=dataset.column_names)

# checagem de tamanho ANTES de treinar — se algum exemplo estourar
# TAMANHO_MAX_SEQUENCIA, ele fica cortado (perde pedaço da conversa ou das
# ferramentas), o que é ruim. Melhor saber agora que descobrir depois.
tamanhos = [len(tokenizer(t)["input_ids"]) for t in dataset["text"]]
print(f"Tamanho dos exemplos em tokens — min: {min(tamanhos)}, máx: {max(tamanhos)}, média: {sum(tamanhos)/len(tamanhos):.0f}")
estourando = sum(1 for t in tamanhos if t > TAMANHO_MAX_SEQUENCIA)
if estourando:
    print(f"⚠️  {estourando} exemplo(s) passam de {TAMANHO_MAX_SEQUENCIA} tokens e serão CORTADOS. Considere aumentar TAMANHO_MAX_SEQUENCIA.")
else:
    print("Todos os exemplos cabem dentro do limite — nenhum corte.")

print()
print("Exemplo formatado (primeiros 800 caracteres):")
print(dataset[0]["text"][:800])
print()

# ---------------------------------------------------------------------------
# 4) Treino
# ---------------------------------------------------------------------------
config_treino = SFTConfig(
    output_dir=PASTA_SAIDA,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    max_length=TAMANHO_MAX_SEQUENCIA,  # TRL renomeou de max_seq_length pra max_length
    dataset_text_field="text",
    gradient_checkpointing=True,  # sequências longas (média 5.2k tokens) — isso poupa bastante VRAM na T4
    gradient_checkpointing_kwargs={"use_reentrant": False},  # recomendação atual pra PEFT + 4-bit
    optim="paged_adamw_8bit",  # otimizador que evita picos de memória em sequência longa
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=config_treino,
    train_dataset=dataset,
)

print("Iniciando treino...")
trainer.train()

print("Salvando adaptador LoRA...")
trainer.save_model(PASTA_SAIDA)
tokenizer.save_pretrained(PASTA_SAIDA)
print(f"Pronto! Adaptador salvo em {PASTA_SAIDA}")
print("Lembra de copiar essa pasta pro Google Drive antes de fechar o Colab — ele apaga tudo quando a sessão cai.")
