# Fine-Tuning di TinyLlama-1.1B-Chat-v1.0

Questo notebook mostra un esempio pratico di **Fine-Tuning supervisionato (SFT)** su un modello di linguaggio di grandi dimensioni (**LLM**) partendo da un **modello base pre-addestrato**, in questo caso:

> **[TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)**

L’obiettivo del progetto è **replicare un processo completo di fine-tuning instruction-following** utilizzando dataset open source e tecniche di ottimizzazione a basso costo computazionale come **LoRA (Low-Rank Adaptation)**.

---

## Base Model

| Proprietà | Dettaglio |
|------------|------------|
| **Modello** | TinyLlama-1.1B-Chat-v1.0 |
| **Parametri** | ~1.1 miliardo |
| **Architettura** | Transformer Decoder-only (GPT-like) |
| **Framework** | Hugging Face Transformers + TRL + PEFT |
| **Obiettivo** | Adattamento instruction-following (SFT) |
| **Motivazione** | Modello open e leggero, gestibile su GPU locali |

---

## Dataset di Addestramento

- **Nome:** [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)  
- **Dimensione:** ~15.000 esempi  
- **Formato:** JSON / HuggingFace Dataset  
- **Contenuto:** prompt di tipo *instruction-following* con campi:
  - `instruction`: richiesta testuale  
  - `context`: contesto opzionale  
  - `response`: risposta ideale del modello  

**Esempio:**
```json
{
  "instruction": "Spiega la differenza tra machine learning e deep learning.",
  "context": "",
  "response": "Il machine learning usa algoritmi per imparare dai dati. Il deep learning utilizza reti neurali con più strati per rappresentazioni più complesse."
}
````

Durante il preprocessing, `instruction` e `context` vengono concatenati per formare il testo di input, mentre `response` è usato come target supervisionato.

---

## Pipeline di Fine-Tuning

### 1. Tokenizzazione

* Tokenizer nativo TinyLlama.
* Padding dinamico, truncation a `max_seq_length=1024`.
* Data Collator:

  ```python
  from transformers import DataCollatorForLanguageModeling
  collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
  ```

### 2. Modello e LoRA

* Backbone congelato.
* Adattamento tramite **LoRA (Low-Rank Adaptation)**:

  ```python
  from peft import LoraConfig
  lora_cfg = LoraConfig(
      task_type="CAUSAL_LM",
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      target_modules=["q_proj", "v_proj", "o_proj", "k_proj"],
      bias="none"
  )
  ```

### 3. Training con TRL

* Framework: `trl==0.23.1`
* Trainer:

  ```python
  from trl import SFTTrainer

  trainer = SFTTrainer(
      model=model,
      peft_config=lora_cfg,
      train_dataset=ds["train"].select(range(5000)),
      data_collator=collator,
      processing_class=tokenizer,
      args=TrainingArguments(
          output_dir=str(OUT_DIR),
          per_device_train_batch_size=2,
          gradient_accumulation_steps=8,
          learning_rate=2e-4,
          num_train_epochs=1,
          fp16=True,
          gradient_checkpointing=True,
          lr_scheduler_type="cosine",
          warmup_ratio=0.03,
          logging_steps=50,
          save_steps=500,
          report_to="none"
      )
  )

  trainer.train()
  ```

---

## 🧱 Architettura del Modello

```
┌────────────────────────────────────┐
│ TinyLlama 1.1B (Base)             │
│ ───────────────────────────────── │
│  Transformer Decoder-only         │
│  24 Layer Blocks                  │
│  16 Attention Heads               │
│  Rotary Embeddings (RoPE)         │
│  Context length: 2048             │
└────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────┐
│ LoRA Adapters                     │
│ ───────────────────────────────── │
│ Applied to Q, K, V, O projections │
│ Trainable rank: 16                │
│ Dropout: 0.05                     │
└────────────────────────────────────┘
           │
           ▼
    Fine-tuned Model
```

---

## Hardware & Ambiente

| Componente     | Dettaglio                            |
| -------------- | ------------------------------------ |
| **GPU**        | NVIDIA RTX A3000 (12 GB VRAM)        |
| **CUDA**       | 12.7                                 |
| **PyTorch**    | 2.4+ (CUDA build)                    |
| **Python**     | 3.11                                 |
| **Frameworks** | Transformers · TRL · PEFT · Datasets |
| **Ambiente**   | venv                         |

---

## Esecuzione Rapida

Clona la repo e avvia il notebook:

```bash
# 1. Clona il repository
git clone https://github.com/<tuo-username>/DeepLearning_MachineLearning.git
cd DeepLearning_MachineLearning/FineTuning_TinyLlama

# 2. Crea ambiente virtuale
python -m venv venv
venv\Scripts\activate      # (Windows)
# oppure source venv/bin/activate (Linux/Mac)

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Avvia Jupyter
jupyter notebook

# 5. Apri il file FineTuning_TinyLlama.ipynb ed esegui le celle
```

---

## Metriche di Valutazione

| Metrica                      | Descrizione                                | Obiettivo                                  |
| ---------------------------- | ------------------------------------------ | ------------------------------------------ |
| **Training Loss**            | Cross-Entropy Loss durante l’addestramento | Deve diminuire progressivamente            |
| **Validation Loss**          | Loss su dataset non visto                  | Misura overfitting                         |
| **Perplexity**               | ( e^{loss} )                               | Valuta la probabilità di sequenze testuali |
| **BLEU / ROUGE (opzionale)** | Confronta testo generato con target        | Misura qualità linguistica                 |
| **GPU Memory (VRAM)**        | Monitorata con nvidia-smi                  | Stabilità durante training                 |

---

## Risultati Sperimentali

* Addestramento su subset di 5.000 esempi.
* Training stabile con `loss ≈ 2.1` dopo 1 epoca.
* Output più coerenti e naturali nel rispetto dell’istruzione.

**Esempio:**

> **Input:**
> “Cos'è l'intelligenza artificiale?”
>
> **Output dopo fine-tuning:**
> “È la capacità delle macchine di apprendere, ragionare e prendere decisioni in modo autonomo, imitando alcuni processi cognitivi umani.”

---

## Salvataggio e Caricamento

```python
trainer.save_model("tinyllama-dolly-lora")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(model, "tinyllama-dolly-lora")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

---

## Obiettivi del Progetto

* ✅ Replicare un fine-tuning supervisionato su LLM.
* ✅ Sperimentare PEFT/LoRA su GPU consumer.
* ✅ Gestire dataset *instruction-following*.
* ✅ Creare pipeline riproducibile, con documentazione completa.
* ✅ Preparare base per esperimenti successivi (QLoRA, RLHF).

---

## 🔍 Visualizzazione del Workflow

```
Dataset (Dolly 15K)
       │
       ▼
Preprocessing → Tokenizzazione
       │
       ▼
Base Model (TinyLlama-1.1B)
       │
       ▼
+ LoRA Adapters (PEFT)
       │
       ▼
Fine-Tuning (SFTTrainer - TRL)
       │
       ▼
Model Evaluation → Salvataggio → Inference
```

---

## Riferimenti

* [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
* [Databricks Dolly 15k Dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
* [Hugging Face TRL – SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
* [PEFT – LoRA](https://huggingface.co/docs/peft/index)
* [Transformers Docs](https://huggingface.co/docs/transformers/index)

---

## Autore

**Alessandro Laudisa**
Machine Learning & AI Engineer
*DeepLearning_MachineLearning – Fine-Tuning Experiments*

- [laudisa2@gmail.com](mailto:laudisa2@gmail.com)

---