from openai import OpenAI

base_url = ""
client = OpenAI(base_url=base_url, api_key="EMPTY")
system = '你是一个人工智能助手。'
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "如何计算圆周率前10位。请一步一步思考，不少于2000字。"}
]
response1 = client.chat.completions.create(
    model="Qwen2.5-72B-Instruct-2",
    messages=messages,
    logprobs=True,
    n=1,
    extra_body={
        "prompt_logprobs": 1
    }
)
prompt_len = len(response1.prompt_logprobs)

messages.append({
    "role": "assistant",
    "content": response1.choices[0].message.content
})
response2 = client.chat.completions.create(
    model="Qwen2.5-72B-Instruct-2",
    messages=messages,
    logprobs=True,
    n=1,
    max_tokens=1,
    extra_body={
        "prompt_logprobs": 1,
    }
)

token_logprobs1 = [(token.token, token.logprob) for token in response1.choices[0].logprobs.content]
s1 = 0
s2 = 0
res2 = response2.prompt_logprobs[prompt_len:]
for idx, tp in enumerate(token_logprobs1):
    for token in res2[idx]:
        if res2[idx][token]['decoded_token'] == tp[0]:
            print([tp[0]], tp[1], res2[idx][token]['logprob'])
            s1+=tp[1]
            s2+=res2[idx][token]['logprob']
print(s1, s2)