# InterpretableQA-LLMTools

Adapted repo structure from ToolQA https://github.com/night-chen/ToolQA

- [ ] Run ```pip install -r requirements.txt``` (make sure openai version is 0.28.0 or earlier)
- [ ] Download data
- [ ] Run preprocessing/data_loading.ipynb
- [ ] Run python scripts in dataset_generation/easy_questions dataset_generation/medium_questions dataset_generation/hard_questions
- [ ] Set openai api key environment variable by ```export OPENAI_API_KEY="[YOUR_OPENAI_API_KEY]"``` (make sure you copy & paste the correct quotation mark written here)
- [ ] Run ```bash benchmark/chameleon/run_toolqa/scripts/run_chameleon.sh```

```easy```: can be solved with interpretable tools with 100% accuracy <br>
```medium```: can be solved by both interpretable tools and non-interpretable tools with comparable performance <br>
```hard```: can be only solved with non-interpretable tools or non-interpretable tools lead to much better performance compared to interpretable tools
