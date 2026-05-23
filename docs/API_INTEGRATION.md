# SigerLM API Integration

SigerLM exposes a FastAPI server for web and mobile apps. The API supports normal JSON responses, Server-Sent Events streaming, session memory, tool-result context, and user feedback collection.

## Run Server

```powershell
python serve_api.py --checkpoint checkpoints\lora\model_t4x2_gpu_stage2_balanced_merged.pt --host 0.0.0.0 --port 8000
```

Optional environment variables:

```powershell
$env:SIGER_API_KEY="change-this-key"
$env:SIGER_CORS_ORIGINS="http://localhost:3000,https://your-app.com"
```

If `SIGER_API_KEY` is set, clients must send:

```txt
X-Siger-API-Key: change-this-key
```

For public deployment, put SigerLM behind HTTPS, a reverse proxy, and provider-level abuse protection. Keep detailed security review notes private.

## Status

```bash
curl http://localhost:8000/v1/status
```

## Generate

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Jelaskan singkat apa itu machine learning.","max_new_tokens":120}'
```

## Chat Session

`session_id` is optional. If omitted, the server creates one.

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user-123","message":"Halo, siapa kamu?","max_new_tokens":120}'
```

## Streaming Chat

```bash
curl -N -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user-123","message":"Tulis ringkasan pendek tentang Lampung O.","stream":true}'
```

Browser example:

```js
const response = await fetch("http://localhost:8000/v1/chat", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-Siger-API-Key": "change-this-key"
  },
  body: JSON.stringify({
    session_id: "user-123",
    message: "Jelaskan apa itu LoRA.",
    stream: false
  })
});

const data = await response.json();
console.log(data.response);
```

## Add Memory

```bash
curl -X POST http://localhost:8000/v1/chat/user-123/memory/document \
  -H "Content-Type: application/json" \
  -d '{"text":"SigerLM adalah model eksperimen trilingual Indonesia, English, dan Lampung O.","metadata":{"source":"app_profile"}}'
```

## Add Tool Result

```bash
curl -X POST http://localhost:8000/v1/chat/user-123/memory/tool-result \
  -H "Content-Type: application/json" \
  -d '{"command":"git diff","output":"diff --git ...","metadata":{"source":"mobile_debug"}}'
```

Tool results are compressed before being stored in memory.

## Feedback

Thumbs up/down or stars:

```bash
curl -X POST http://localhost:8000/v1/feedback/rating \
  -H "Content-Type: application/json" \
  -d '{"session_id":"user-123","prompt":"Apa itu LoRA?","response":"LoRA adalah...","rating":5,"category":"general"}'
```

Reviewed preference for DPO:

```bash
curl -X POST http://localhost:8000/v1/feedback/preference \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Terjemahkan Nyak haga mengan.","chosen_response":"Saya mau makan.","rejected_response":"Saya pergi.","rating":5,"category":"lampung","approved_for_training":true}'
```

Keep `approved_for_training=false` for raw user feedback until a human has reviewed it.
