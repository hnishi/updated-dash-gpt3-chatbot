# v1.x
from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = "08ef6481c89c485cba4c9ecae08c7370",  
  api_version = "2023-05-15",
  azure_endpoint ="https://openaiexperimentsinstance.openai.azure.com/"
)

response = client.chat.completions.create(
    model="gpt35-16k", #"gpt-35-turbo", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        {"role": "user", "content": "Who were the founders of Microsoft?"}
    ]
)

#print(response)
print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)
