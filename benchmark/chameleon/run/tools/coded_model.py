import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define constants
text = 'We propose a Bayesian encoder for metric learning...'
section = 'Abstract'
target = 'Poster Session'
unique_classes = ["not 2", "2"]
CLASSES = 2
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure model and tokenizer
config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=CLASSES)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=max_length)
model = AutoModelForSequenceClassification.from_config(config=config).to(device)

# Tokenize input text
def tokenize_text(text, tokenizer, max_length):
    return tokenizer(text, truncation=True, padding='max_length', max_length=max_length)

encoded_text = tokenize_text(text, tokenizer, max_length)
input_ids = torch.tensor(encoded_text['input_ids']).unsqueeze(0).to(device)
attention_mask = torch.tensor(encoded_text['attention_mask']).unsqueeze(0).to(device)
processed_text = {'input_ids': input_ids, 'attention_mask': attention_mask}

# DataLoader setup for training
def prepare_data(df, section, tokenizer, target, batch_size=64):
    zero_encoding = tokenize_text('', tokenizer, max_length)  # Now it correctly references the function
    df[section] = df[section].apply(lambda text: tokenize_text(text, tokenizer, max_length) if text is not None else zero_encoding)
    df['input_ids'] = df[section].apply(lambda x: torch.tensor(x['input_ids']))
    df['attention_mask'] = df[section].apply(lambda x: torch.tensor(x['attention_mask']))
    df['output'] = df[target].apply(lambda x: int(x == "2"))  # Simplified binary mapping
    dataset = df[['input_ids', 'attention_mask', 'output']].apply(
        lambda row: {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'], 'output': row['output']}, axis=1
    )
    return DataLoader(list(dataset), batch_size=batch_size)

# Train the model
def train_model(model, data_loader, epochs=5, learning_rate=2e-5):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(data_loader):
            inputs, labels = batch['input_ids'].to(device), batch['output'].to(device)
            outputs = model(input_ids=inputs, labels=labels).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# Evaluate the model on the input text
def predict(model, processed_text, unique_classes):
    model.eval()
    inputs = processed_text['input_ids']
    with torch.no_grad():
        outputs = model(input_ids=inputs).logits
        prediction = torch.argmax(outputs, dim=1).item()
    return {"prediction": unique_classes[prediction]}

ans = predict(model, processed_text, unique_classes)