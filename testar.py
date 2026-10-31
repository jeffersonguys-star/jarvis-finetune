import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen3-4B"
ADAPTADOR = "jarvis-qwen3-lora"
tok = AutoTokenizer.from_pretrained(ADAPTADOR)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
modelo = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
modelo = PeftModel.from_pretrained(modelo, ADAPTADOR)
modelo.eval()

primeira = json.loads(open("dataset_jarvis.jsonl", encoding="utf8").readline())
por_nome = {t["function"]["name"]: t for t in primeira["tools"]}
todas = {}
for l in open("dataset_jarvis.jsonl", encoding="utf8"):
    for t in json.loads(l)["tools"]:
        todas[t["function"]["name"]] = t
for n in ["criar_canal", "adicionar_cargo_membro", "enviar_mensagem", "deletar_canal", "banir_membro"]:
    if n in todas:
        por_nome[n] = todas[n]
tools = list(por_nome.values())

SYSTEM = ("Voce e Jarvis, um assistente de administracao de servidores Discord. "
          "Responda ao pedido do usuario usando as ferramentas disponiveis quando necessario. "
          "Se faltar informacao pra usar uma ferramenta, pergunte antes de chamar. "
          "Se nao existir ferramenta pra algo, diga isso claramente em vez de inventar. "
          "Nunca invente um fato que nenhuma ferramenta confirmou.")

PEDIDOS = [
    "abre uma sala de voz chamada Reunião",
    "bota o cargo Staff no Pedro",
    "manda um oi no canal 1538443057823748159",
    "cria um canal",
    "qual a previsão do tempo amanhã?",
    "apaga o canal 999 e bane o Zeca",
    "vai chover hoje?",
    "me conta uma piada",
    "bane o Zeca por spam",
    "quanto é 7 vezes 8?",
]

for p in PEDIDOS:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
    txt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                  tokenize=False, enable_thinking=False)
    ids = tok(txt, return_tensors="pt").to(modelo.device)
    with torch.no_grad():
        out = modelo.generate(**ids, max_new_tokens=200, do_sample=False)
    resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    print("\n=== PEDIDO:", p, "\n", resp.replace("<|im_end|>", "").strip(), flush=True)
