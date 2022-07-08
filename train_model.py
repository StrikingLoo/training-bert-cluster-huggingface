from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM



import torch

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=30,
    max_position_embeddings=5000,
    num_attention_heads=32,
    num_hidden_layers=6,
    type_vocab_size=1,
)


tokenizer = RobertaTokenizerFast.from_pretrained(".", max_len=5000)



model = RobertaForMaskedLM(config=config)

model.num_parameters()

###### Model is set up! Now dataset

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./sequences_train.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./",
    overwrite_output_dir=True,
    num_train_epochs=120,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./model_32_6")


########################################### POST TRAIN!! ####

#trainer.save_model("./model_all")


#from transformers import pipeline

#fill_mask = pipeline(
#    "fill-mask",
#    model="./model_all",
#    tokenizer=tokenizer
#)

#fill_mask("TGYVSTMPKVIIFTDFDGTVTGKSGNETVFTEFYQSLLQGYK<mask>DVEQDYKNTPMKDPIEAQALFEAKYGKYNENFDHDQQDVDFLMSPEAVAFFHEVLKNDDVTVNIVTKNRAEYIKAVFKYQGFSDEEISKLTILESGYKFNDVNSRLNHPTERANRVYILDDSPTDYAEMLRAVKGKGYNEEEIRGYRKNPGEFEWSQYLEDVREMFPPKEN")

#tokenizer = ByteLevelBPETokenizer(
#    "./vocab.json",
#    "./merges.txt",
#)

#tokenizer._tokenizer.post_processor = BertProcessing(
#    ("</s>", tokenizer.token_to_id("</s>")),
#    ("<s>", tokenizer.token_to_id("<s>")),
#)
#tokenizer.enable_truncation(max_length=5000)
